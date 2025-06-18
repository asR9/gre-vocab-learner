from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import random
import json
from database import (
    sample_words_for_quiz,
    update_word_status,
    get_words_by_status,
    save_session_data,
    get_word_definition,
    batch_update_word_statuses_session_end,
    save_note,
    get_notes_for_word,
    get_all_notes,
    get_example_sentence,
    save_example_sentence
)
from intent_classifier import IntentClassifier
from rag_system import RAGSystem

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_mode: str
    session_quiz_performance: Dict[str, Any]
    generated_mnemonics: Dict[str, str]
    user_notes: Dict[str, str]
    current_quiz_words: List[tuple]
    waiting_for_quiz_answers: bool
    waiting_for_hint: bool
    last_user_input: str
    session_active: bool
    pending_updates: Dict[str, bool]  # For session-end batch updates
    conversation_context: List[Dict[str, str]]

class GREVocabAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.intent_classifier = IntentClassifier()
        # Initialize RAG system
        try:
            self.rag_system = RAGSystem()
            print("✓ RAG system initialized")
        except Exception as e:
            print(f"Warning: RAG system initialization failed: {e}")
            self.rag_system = None

        self.state = AgentState(
            messages=[],
            current_mode="router",
            session_quiz_performance={
                "words_attempted": [],
                "correct_answers": [],
                "incorrect_answers": [],
                "weak_words": [],
                "session_score": 0.0
            },
            generated_mnemonics={},
            user_notes={},
            current_quiz_words=[],
            waiting_for_quiz_answers=False,
            waiting_for_hint=False,
            last_user_input="",
            session_active=False,
            pending_updates={},
            conversation_context=[]
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("start_quiz", self._quiz_node)
        workflow.add_node("study_words", self._study_node)
        workflow.add_node("add_note", self._notes_node)
        workflow.add_node("view_notes", self._notes_node)
        workflow.add_node("summarize_notes", self._summarize_notes_node)
        workflow.add_node("get_hints", self._hints_node)
        workflow.add_node("end_session", self._end_session_node)
        workflow.add_node("search_words", self._search_words_node)
        workflow.add_node("get_progress", self._progress_node)
        workflow.add_node("general", self._general_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "start_quiz": "start_quiz",
                "study_words": "study_words",
                "add_note": "add_note",
                "view_notes": "view_notes",
                "summarize_notes": "summarize_notes",
                "get_hints": "get_hints",
                "end_session": "end_session",
                "search_words": "search_words",
                "get_progress": "get_progress",
                "general": "general",
                "end": END
            }
        )

        # All nodes end the workflow
        for node in ["start_quiz", "study_words", "add_note", "view_notes", 
                    "summarize_notes", "get_hints", "end_session", "search_words", 
                    "get_progress", "general"]:
            workflow.add_edge(node, END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        user_input = state["last_user_input"]

        # Handle special cases first
        if state["waiting_for_quiz_answers"]:
            if "hint" in user_input.lower():
                state["current_mode"] = "get_hints"
                state["waiting_for_hint"] = True
            else:
                state["current_mode"] = "start_quiz"  # Process quiz answers
            return state

        # Use LLM-based intent classification
        try:
            intent_result = self.intent_classifier.classify_intent(
                user_input, 
                state["conversation_context"]
            )
            state["current_mode"] = intent_result["intent"]
        except Exception as e:
            print(f"Intent classification error: {e}")
            state["current_mode"] = "general"

        return state

    def _route_decision(self, state: AgentState) -> str:
        return state["current_mode"]

    def _quiz_node(self, state: AgentState) -> AgentState:
        # If we're waiting for answers, process them
        if state["waiting_for_quiz_answers"]:
            return self._process_quiz_answers(state)

        # Otherwise, generate new quiz
        return self._generate_new_quiz(state)

    def _generate_new_quiz(self, state: AgentState) -> AgentState:
        user_input = state["last_user_input"]

        # Extract quiz parameters using LLM
        try:
            quiz_params = self.intent_classifier.extract_quiz_parameters(user_input)
        except:
            quiz_params = {"word_count": 5, "specific_words": [], "difficulty": None, "focus_status": None}

        # Start session if not active
        if not state["session_active"]:
            state["session_active"] = True
            state["pending_updates"] = {}

        # Get quiz words based on parameters
        if quiz_params["specific_words"]:
            from database import search_words
            quiz_words = []
            for word in quiz_params["specific_words"]:
                matches = search_words(word, 1)
                if matches:
                    quiz_words.append(matches[0])
        elif quiz_params["focus_status"]:
            quiz_words = get_words_by_status(quiz_params["focus_status"], quiz_params["word_count"])
        else:
            # Smart selection: prioritize unknown/weak words
            unknown_words = get_words_by_status("unknown", quiz_params["word_count"] // 2)
            weak_words = get_words_by_status("weak", quiz_params["word_count"] // 2)
            remaining = quiz_params["word_count"] - len(unknown_words) - len(weak_words)
            
            if remaining > 0:
                other_words = sample_words_for_quiz(remaining)
                quiz_words = unknown_words + weak_words + other_words
            else:
                quiz_words = unknown_words + weak_words

        if not quiz_words:
            response = "Sorry, couldn't generate quiz. Database might be empty."
            state["messages"].append({"role": "assistant", "content": response})
            return state

        # Limit to requested count
        quiz_words = quiz_words[:quiz_params["word_count"]]
        state["current_quiz_words"] = quiz_words
        state["waiting_for_quiz_answers"] = True

        # Create enhanced quiz with better formatting
        response = f"🧠 **Quiz Time!** ({len(quiz_words)} words)\n\n"
        for i, (word, _) in enumerate(quiz_words, 1):
            response += f"**{i}. {word.upper()}**\n"
        
        response += f"\n💡 **Instructions:**\n"
        response += f"• Provide definitions for each word\n"
        response += f"• You can number them or just list them\n"
        response += f"• Type 'hint' if you need help with any word\n"
        response += f"• Session will save progress when you're done\n"

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _process_quiz_answers(self, state: AgentState) -> AgentState:
        user_answers = state["last_user_input"]
        quiz_words = state["current_quiz_words"]

        # Grade answers using enhanced LLM grading
        results, detailed_feedback = self._grade_answers_enhanced(quiz_words, user_answers)

        # Create comprehensive feedback
        feedback = "📊 **Quiz Results:**\n\n"
        
        for word, correct_def in quiz_words:
            is_correct = results.get(word, False)
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            
            feedback += f"**{word.upper()}**: {status}\n"
            feedback += f"*Correct Definition:* {correct_def}\n"
            
            # Add detailed feedback from LLM
            if word in detailed_feedback:
                feedback += f"*Feedback:* {detailed_feedback[word]}\n"
            
            # Add example sentence if available
            example = get_example_sentence(word)
            if not example:
                example = self._generate_example_sentence(word, correct_def)
                if example:
                    save_example_sentence(word, example)
            
            if example:
                feedback += f"*Example:* {example}\n"
            
            feedback += "\n"

        # Update performance tracking
        correct_count = sum(results.values())
        total_count = len(results)
        session_score = correct_count / total_count if total_count > 0 else 0

        state["session_quiz_performance"]["words_attempted"].extend([w for w, _ in quiz_words])
        state["session_quiz_performance"]["correct_answers"].extend([w for w, correct in results.items() if correct])
        state["session_quiz_performance"]["incorrect_answers"].extend([w for w, correct in results.items() if not correct])
        state["session_quiz_performance"]["weak_words"].extend([w for w, correct in results.items() if not correct])
        state["session_quiz_performance"]["session_score"] = session_score

        # Store results for session-end batch update
        state["pending_updates"].update(results)

        # Performance summary with encouragement
        feedback += f"📈 **Session Score: {session_score:.1%}** ({correct_count}/{total_count})\n"
        
        if session_score >= 0.8:
            feedback += "🎉 Excellent work! You're mastering these words!\n"
        elif session_score >= 0.6:
            feedback += "👍 Good progress! Keep practicing the challenging ones.\n"
        else:
            feedback += "💪 Don't worry! Review the definitions and try again.\n"

        feedback += "\n**What's next?**\n"
        feedback += "• 'quiz' - Take another quiz\n"
        feedback += "• 'study' - Review difficult words\n"
        feedback += "• 'end session' - Save progress and finish\n"

        # Reset quiz state
        state["current_quiz_words"] = []
        state["waiting_for_quiz_answers"] = False

        state["messages"].append({"role": "assistant", "content": feedback})
        return state

    def _grade_answers_enhanced(self, quiz_words: List[tuple], user_answers: str) -> tuple[Dict[str, bool], Dict[str, str]]:
        """Grade answers using enhanced LLM with detailed feedback"""
        results = {}
        detailed_feedback = {}

        # Create a comprehensive grading prompt for all words at once
        words_info = ""
        for i, (word, correct_def) in enumerate(quiz_words, 1):
            words_info += f"{i}. {word} = {correct_def}\n"

        prompt = f"""
        Grade the user's quiz answers and provide detailed feedback.

        Quiz words and correct definitions:
        {words_info}

        User's answers: "{user_answers}"

        For each word, determine if the user's answer shows understanding of the core meaning.
        Be generous - if they capture the essence, even with different wording, mark as correct.

        Respond in JSON format:
        {{
            "word1": {{
                "correct": true/false,
                "feedback": "Brief explanation of why correct/incorrect and encouragement"
            }},
            "word2": {{
                "correct": true/false,
                "feedback": "Brief explanation..."
            }}
        }}

        Make feedback encouraging and educational. For incorrect answers, gently explain the correct meaning.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            grading_result = json.loads(response.content.strip())
            
            for word, _ in quiz_words:
                word_key = word.lower()
                if word_key in grading_result:
                    results[word] = grading_result[word_key]["correct"]
                    detailed_feedback[word] = grading_result[word_key]["feedback"]
                else:
                    # Fallback to individual grading
                    results[word] = self._grade_single_answer_llm(word, _, user_answers)
                    detailed_feedback[word] = "Good effort!"
        except Exception as e:
            print(f"Enhanced grading error: {e}")
            # Fallback to simple grading
            for word, correct_def in quiz_words:
                results[word] = self._grade_single_answer_llm(word, correct_def, user_answers)
                detailed_feedback[word] = "Keep practicing!"

        return results, detailed_feedback

    def _grade_answers(self, quiz_words: List[tuple], user_answers: str) -> Dict[str, bool]:
        """Grade answers using LLM only (legacy method)"""
        results = {}

        # Use LLM for all grading
        for word, correct_def in quiz_words:
            results[word] = self._grade_single_answer_llm(word, correct_def, user_answers)

        return results

    def _grade_single_answer_llm(self, word: str, correct_def: str, user_answers: str) -> bool:
        """Grade a single answer using LLM"""
        prompt = f"""
        The word is "{word}" and the correct definition is "{correct_def}".

        The user provided these answers: "{user_answers}"

        Does any part of the user's answer show they understand what "{word}" means?
        Be generous - if they capture the core meaning, even with different words, that's correct.

        Examples of correct understanding:
        - "illicit" = "forbidden by law" should match "illegal", "against the law", "prohibited", "unlawful"
        - "sophomoric" = "conceited and overconfident" should match "arrogant", "pretentious", "showing off knowledge"

        Answer only "YES" or "NO".
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return "yes" in response.content.lower()

    def _generate_example_sentence(self, word: str, definition: str) -> str:
        """Generate an example sentence for a word"""
        prompt = f"""
        Create a clear, engaging example sentence for the GRE word "{word}" meaning "{definition}".
        
        Requirements:
        - Use the word naturally in context
        - Make the meaning clear from context
        - Keep it concise (under 20 words)
        - Make it memorable and relatable
        
        Just return the sentence, nothing else.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error generating example: {e}")
            return None

    def _hints_node(self, state: AgentState) -> AgentState:
        """Provide hints for quiz words"""
        if not state["current_quiz_words"]:
            response = "No active quiz to provide hints for. Start a quiz first!"
            state["messages"].append({"role": "assistant", "content": response})
            return state

        response = "💡 **Hints for current quiz:**\n\n"
        
        for i, (word, definition) in enumerate(state["current_quiz_words"], 1):
            # Generate progressive hints
            first_letter = word[0].upper()
            word_length = len(word)
            
            # Try to get a synonym or related word
            hint_word = "similar concept"
            if self.rag_system:
                try:
                    related_words = self.rag_system.find_related_words(word, 2)
                    hint_word = related_words[0]["word"] if related_words else "similar concept"
                except:
                    pass
            
            response += f"**{i}. {word.upper()}**\n"
            response += f"• Starts with '{first_letter}' ({word_length} letters)\n"
            response += f"• Related to: {hint_word}\n"
            response += f"• Think about: {definition.split(',')[0]}...\n\n"

        response += "Ready to answer? Or need more specific hints for any word?"
        
        state["waiting_for_hint"] = False
        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _end_session_node(self, state: AgentState) -> AgentState:
        """End the current session and save all progress"""
        if not state["session_active"]:
            response = "No active session to end. Start a quiz or study session first!"
            state["messages"].append({"role": "assistant", "content": response})
            return state

        # Batch update all word statuses
        if state["pending_updates"]:
            try:
                batch_update_word_statuses_session_end(state["pending_updates"])
                updated_count = len(state["pending_updates"])
            except Exception as e:
                print(f"Error updating word statuses: {e}")
                updated_count = 0
        else:
            updated_count = 0

        # Save session data
        session_data = {
            "session_quiz_performance": state["session_quiz_performance"],
            "generated_mnemonics": state["generated_mnemonics"],
            "user_notes": state["user_notes"],
            "session_type": "mixed",
            "notes_created": len(state["user_notes"])
        }
        
        try:
            save_session_data(session_data)
        except Exception as e:
            print(f"Error saving session: {e}")

        # Generate session summary
        perf = state["session_quiz_performance"]
        total_attempted = len(perf["words_attempted"])
        total_correct = len(perf["correct_answers"])
        session_score = perf.get("session_score", 0)

        response = "🎯 **Session Complete!**\n\n"
        response += f"📊 **Final Stats:**\n"
        response += f"• Words Attempted: {total_attempted}\n"
        response += f"• Correct Answers: {total_correct}\n"
        response += f"• Session Score: {session_score:.1%}\n"
        response += f"• Word Statuses Updated: {updated_count}\n"
        response += f"• Notes Created: {len(state['user_notes'])}\n\n"

        if session_score >= 0.8:
            response += "🏆 Outstanding performance! You're making excellent progress.\n"
        elif session_score >= 0.6:
            response += "👍 Good work! Keep practicing to improve further.\n"
        else:
            response += "💪 Every practice session helps! Review and try again.\n"

        response += "\n**Ready for your next session?** Just say 'start quiz' or 'study words'!"

        # Reset session state
        state["session_active"] = False
        state["pending_updates"] = {}
        state["current_quiz_words"] = []
        state["waiting_for_quiz_answers"] = False

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _summarize_notes_node(self, state: AgentState) -> AgentState:
        """Summarize user notes using RAG"""
        user_input = state["last_user_input"]
        
        # Extract query if user specified what to summarize
        query = None
        if "about" in user_input.lower() or "on" in user_input.lower():
            words = user_input.lower().split()
            try:
                if "about" in words:
                    query_start = words.index("about") + 1
                elif "on" in words:
                    query_start = words.index("on") + 1
                query = " ".join(words[query_start:])
            except:
                pass

        if self.rag_system:
            try:
                summary = self.rag_system.summarize_notes(query)
                response = f"📝 **Notes Summary:**\n\n{summary}"
                
                if query:
                    response += f"\n\n🔍 *Filtered by: {query}*"
            except Exception as e:
                print(f"Error summarizing notes: {e}")
                response = "Sorry, I couldn't summarize your notes right now. Try viewing them directly with 'show notes'."
        else:
            response = "Note summarization requires OpenAI API access. You can still view your notes with 'show notes'."

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _search_words_node(self, state: AgentState) -> AgentState:
        """Search for words using RAG similarity"""
        user_input = state["last_user_input"]
        
        # Extract search query
        search_terms = ["find", "search", "like", "similar"]
        query = user_input
        for term in search_terms:
            if term in user_input.lower():
                parts = user_input.lower().split(term)
                if len(parts) > 1:
                    query = parts[1].strip()
                break

        if self.rag_system:
            try:
                # Search for similar words
                similar_words = self.rag_system.search_similar_words(query, 5)
                
                if similar_words:
                    response = f"🔍 **Words similar to '{query}':**\n\n"
                    for word_info in similar_words:
                        word = word_info["word"]
                        definition = word_info["definition"]
                        similarity = word_info["similarity"]
                        response += f"**{word.upper()}** (similarity: {similarity:.1%})\n"
                        response += f"*Definition:* {definition}\n\n"
                else:
                    response = f"No similar words found for '{query}'. Try a different search term."
            except Exception as e:
                print(f"Error searching words: {e}")
                response = "Sorry, I couldn't search for words right now. Try again later."
        else:
            response = "Semantic word search requires OpenAI API access. You can browse words by taking quizzes or studying."

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _progress_node(self, state: AgentState) -> AgentState:
        """Show user progress and statistics"""
        try:
            from database import get_word_status_counts
            status_counts = get_word_status_counts()
            
            total_words = sum(status_counts.values())
            mastered = status_counts.get('strong', 0)
            learning = status_counts.get('moderate', 0)
            weak = status_counts.get('weak', 0)
            unknown = status_counts.get('unknown', 0)
            
            mastery_rate = mastered / total_words if total_words > 0 else 0
            
            response = "📊 **Your Progress Dashboard:**\n\n"
            response += f"**Overall Stats:**\n"
            response += f"• Total Words: {total_words}\n"
            response += f"• Mastery Rate: {mastery_rate:.1%}\n\n"
            
            response += f"**Word Status Breakdown:**\n"
            response += f"🟢 Strong: {mastered} words\n"
            response += f"🟡 Moderate: {learning} words\n"
            response += f"🟠 Weak: {weak} words\n"
            response += f"⚪ Unknown: {unknown} words\n\n"
            
            # Session stats if available
            if state["session_active"]:
                perf = state["session_quiz_performance"]
                response += f"**Current Session:**\n"
                response += f"• Words Attempted: {len(perf['words_attempted'])}\n"
                response += f"• Correct Answers: {len(perf['correct_answers'])}\n"
                response += f"• Session Score: {perf.get('session_score', 0):.1%}\n\n"
            
            # Recommendations
            if weak > 0:
                response += f"💡 **Recommendation:** Focus on {weak} weak words with 'study weak words'\n"
            elif unknown > 0:
                response += f"💡 **Recommendation:** Explore {unknown} new words with 'quiz unknown words'\n"
            else:
                response += f"🎉 **Great job!** You're doing well. Keep practicing to maintain your progress!\n"
                
        except Exception as e:
            print(f"Error getting progress: {e}")
            response = "Sorry, I couldn't retrieve your progress right now."

        state["messages"].append({"role": "assistant", "content": response})
        return state



    def _study_node(self, state: AgentState) -> AgentState:
        # Get weak words from session or database
        weak_words = state["session_quiz_performance"]["weak_words"]
        if not weak_words:
            weak_from_db = get_words_by_status("weak", 3) + get_words_by_status("unknown", 3)
            weak_words = [word for word, _ in weak_from_db[:5]]

        if not weak_words:
            response = "🎉 Great! No weak words to study. Try taking a quiz to identify areas for improvement!"
            state["messages"].append({"role": "assistant", "content": response})
            return state

        # Generate enhanced study materials
        study_words = weak_words[:3]
        word_definitions = [(word, get_word_definition(word)) for word in study_words]
        mnemonics = self._generate_batch_mnemonics(word_definitions)

        response = "📖 **Enhanced Study Session:**\n\n"
        for i, (word, definition) in enumerate(word_definitions):
            mnemonic = mnemonics[i] if i < len(mnemonics) else f"Remember: {word.upper()} = {definition}"
            state["generated_mnemonics"][word] = mnemonic

            response += f"**{word.upper()}**\n"
            response += f"*Definition:* {definition}\n"
            response += f"*Memory Trick:* {mnemonic}\n"
            
            # Add example sentence
            example = get_example_sentence(word)
            if not example:
                example = self._generate_example_sentence(word, definition)
                if example:
                    save_example_sentence(word, example)
            
            if example:
                response += f"*Example:* {example}\n"
            
            # Add related words using RAG
            if self.rag_system:
                try:
                    related = self.rag_system.find_related_words(word, 2)
                    if related:
                        related_names = [r["word"] for r in related]
                        response += f"*Related words:* {', '.join(related_names)}\n"
                except:
                    pass
            
            # Save mnemonic as note in RAG system
            if self.rag_system:
                try:
                    self.rag_system.add_note_to_rag(word, mnemonic, "mnemonic")
                except:
                    pass
            
            response += "\n"

        response += "💡 **Study Tips:**\n"
        response += "• Review these mnemonics regularly\n"
        response += "• Try using the words in your own sentences\n"
        response += "• Take a quiz to test your memory\n\n"
        response += "Ready for a quiz or want to add your own notes?"
        
        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _generate_simple_mnemonic(self, word: str, definition: str) -> str:
        """Generate creative mnemonics using LLM only"""
        return self._generate_llm_mnemonic(word, definition)

    def _generate_batch_mnemonics(self, word_definitions: List[tuple]) -> List[str]:
        """Generate multiple mnemonics in ONE fast LLM call"""
        if not word_definitions:
            return []

        # Create batch prompt for all words at once
        words_text = ""
        for i, (word, definition) in enumerate(word_definitions, 1):
            words_text += f"{i}. {word.upper()} = {definition}\n"

        # Create dynamic format example
        format_example = ""
        for i in range(1, len(word_definitions) + 1):
            format_example += f"{i}. [mnemonic for word {i}]\n"

        prompt = f"""
        Create memorable mnemonics for these {len(word_definitions)} GRE words. Use sound associations, visual imagery, or word breakdown.

        Words:
        {words_text}

        Format your response exactly as:
        {format_example}
        Keep each mnemonic under 10 words and make them creative!
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Parse the numbered response
        lines = response.content.strip().split('\n')
        mnemonics = []
        for line in lines:
            if line.strip() and any(line.startswith(f"{i}.") for i in range(1, len(word_definitions) + 1)):
                # Remove the number prefix
                mnemonic = line.split('.', 1)[1].strip()
                if mnemonic:  # Only add non-empty mnemonics
                    mnemonics.append(mnemonic)

        return mnemonics[:len(word_definitions)]

    def _generate_llm_mnemonic(self, word: str, definition: str) -> str:
        """Generate mnemonic using LLM"""
        prompt = f"""Create a memorable mnemonic for "{word}" meaning "{definition}".

Use sound-alike words, word breakdown, or visual imagery.

Examples:
- SPARSE = "SPACE is SPARSE" - empty space with scattered stars
- ABATE = "A BAT swinging down" - force reduces
- FACETIOUS = "FACE-TEASE-US" - making inappropriate jokes

Keep under 8 words. Just the mnemonic, no explanation."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()



    def _notes_node(self, state: AgentState) -> AgentState:
        user_input = state["last_user_input"]
        
        # Determine if this is add_note or view_notes based on current_mode
        if state["current_mode"] == "add_note":
            return self._handle_add_note(state)
        else:
            return self._handle_view_notes(state)

    def _handle_add_note(self, state: AgentState) -> AgentState:
        """Handle adding a new note"""
        user_input = state["last_user_input"]
        
        # Extract note information using LLM
        try:
            note_info = self.intent_classifier.extract_note_info(user_input)
        except:
            note_info = {"word": None, "note_content": None, "action": "add"}

        # Check if user provided note in expected format
        if "note for" in user_input.lower() and ":" in user_input:
            try:
                parts = user_input.split(":")
                word_part = parts[0].lower().replace("note for", "").strip()
                note_content = ":".join(parts[1:]).strip()
                
                # Save to session state
                state["user_notes"][word_part] = note_content
                
                # Save to database
                save_note(word_part, note_content, "user")
                
                # Add to RAG system
                if self.rag_system:
                    try:
                        self.rag_system.add_note_to_rag(word_part, note_content, "user")
                    except:
                        pass

                response = f"✅ **Note saved for '{word_part.upper()}':**\n\n"
                response += f"*Your note:* {note_content}\n\n"
                
                # Show definition if available
                definition = get_word_definition(word_part)
                if definition:
                    response += f"*Definition:* {definition}\n\n"
                
                response += "**Add another note or try a different action:**\n"
                response += "• 'note for [word]: [content]' - Add another note\n"
                response += "• 'show notes' - View all your notes\n"
                response += "• 'quiz' - Test your knowledge\n"
                
                state["messages"].append({"role": "assistant", "content": response})
                return state
            except Exception as e:
                print(f"Error saving note: {e}")

        # If format wasn't recognized, provide guidance
        response = """📝 **Add a Note**

To add a note for a word, use this format:
**"note for [word]: [your note]"**

**Examples:**
• "note for sparse: means thin and scattered"
• "note for abate: think of a bat flying away - force reduces"
• "note for facetious: FACE-TEASE-US - inappropriate jokes"

**Tips for good notes:**
• Use memory tricks and associations
• Keep them personal and memorable
• Include visual or sound connections

What word would you like to add a note for?"""

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _handle_view_notes(self, state: AgentState) -> AgentState:
        """Handle viewing existing notes"""
        # Get notes from database and session
        try:
            db_notes = get_all_notes(20)
            all_notes = {}
            
            # Add database notes
            for word, note_content, note_type, timestamp in db_notes:
                if word not in all_notes:
                    all_notes[word] = []
                all_notes[word].append({
                    "content": note_content,
                    "type": note_type,
                    "timestamp": timestamp
                })
            
            # Add session notes
            for word, note in state["user_notes"].items():
                if word not in all_notes:
                    all_notes[word] = []
                all_notes[word].append({
                    "content": note,
                    "type": "user",
                    "timestamp": "current session"
                })
            
            # Add generated mnemonics
            for word, mnemonic in state["generated_mnemonics"].items():
                if word not in all_notes:
                    all_notes[word] = []
                all_notes[word].append({
                    "content": mnemonic,
                    "type": "mnemonic",
                    "timestamp": "current session"
                })

        except Exception as e:
            print(f"Error getting notes: {e}")
            all_notes = {}

        if not all_notes:
            response = """📝 **Your Notes are Empty**

**Get started by:**
• Taking a quiz to generate mnemonics
• Studying words to create memory tricks
• Adding your own notes: "note for [word]: [content]"

**Example:** "note for sparse: remember it means thin or scanty"

What would you like to do? (quiz, study, or add notes)"""
        else:
            response = "📝 **Your Vocabulary Notes:**\n\n"
            
            # Show notes organized by word
            for word, notes in list(all_notes.items())[:10]:  # Show first 10 words
                definition = get_word_definition(word)
                response += f"**{word.upper()}**\n"
                if definition:
                    response += f"*Definition:* {definition}\n"
                
                # Group notes by type
                user_notes = [n for n in notes if n["type"] == "user"]
                mnemonics = [n for n in notes if n["type"] == "mnemonic"]
                
                if user_notes:
                    response += f"*Your notes:* {user_notes[0]['content']}\n"
                if mnemonics:
                    response += f"*Memory trick:* {mnemonics[0]['content']}\n"
                
                response += "\n"

            response += f"**Showing {min(len(all_notes), 10)} of {len(all_notes)} words with notes**\n\n"
            response += "**Actions:**\n"
            response += "• 'note for [word]: [content]' - Add a note\n"
            response += "• 'summarize notes' - Get an AI summary\n"
            response += "• 'find words like [concept]' - Search similar words\n"

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _general_node(self, state: AgentState) -> AgentState:
        response = """🎓 **GRE Vocab Tutor - Enhanced Edition**

I'm here to help you master GRE vocabulary with intelligent features:

**🧠 Quiz Features:**
• "Start a quiz" - Adaptive quiz focusing on weak areas
• "Quiz me on 5 words" - Custom quiz size
• "Quiz weak words" - Target specific difficulty
• "Hint" - Get progressive hints during quizzes

**📖 Study Features:**
• "Study words" - Enhanced study with mnemonics & examples
• "Study weak words" - Focus on challenging vocabulary
• Related word suggestions using AI

**📝 Notes & Memory:**
• "Show notes" - View all your vocabulary notes
• "Note for [word]: [content]" - Add personal notes
• "Summarize notes" - AI-powered note summaries
• Automatic mnemonic generation

**🔍 Smart Search:**
• "Find words like [concept]" - Semantic word search
• "Search similar to sparse" - Discover related vocabulary

**📊 Progress Tracking:**
• "Show progress" - Detailed performance dashboard
• "End session" - Save progress and get summary
• Session-based learning with batch updates

**💡 Example Commands:**
• "Start a quiz with 3 words"
• "Study my weak words"
• "Note for ubiquitous: means everywhere"
• "Find words similar to happy"
• "Show my progress"

What would you like to explore?"""
        
        state["messages"].append({"role": "assistant", "content": response})
        return state

    def process_message(self, user_input: str) -> str:
        """Process user message and return response"""
        self.state["last_user_input"] = user_input
        
        # Update conversation context
        self.state["conversation_context"].append({
            "role": "user", 
            "content": user_input
        })
        
        # Keep only last 10 messages for context
        if len(self.state["conversation_context"]) > 10:
            self.state["conversation_context"] = self.state["conversation_context"][-10:]

        try:
            result = self.graph.invoke(self.state)
            self.state = result

            if self.state["messages"]:
                response = self.state["messages"][-1]["content"]
                
                # Add assistant response to context
                self.state["conversation_context"].append({
                    "role": "assistant",
                    "content": response
                })
                
                return response
            else:
                return "I'm not sure how to help. Try asking about quiz, study, or notes!"
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"Error in process_message: {e}")
            return error_msg

    def save_session_progress(self):
        """Save session progress and end session"""
        if self.state["session_active"]:
            # Trigger end session node
            self.state["current_mode"] = "end_session"
            self.state["last_user_input"] = "end session"
            
            try:
                result = self.graph.invoke(self.state)
                self.state = result
                return "Session saved successfully!"
            except Exception as e:
                print(f"Error saving session: {e}")
                return f"Error saving session: {str(e)}"
        else:
            return "No active session to save."