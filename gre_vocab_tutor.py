from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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
    save_example_sentence,
    get_word_status_counts
)
from rag_system import RAGSystem

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_mode: str
    session_quiz_performance: Dict[str, Any]
    current_quiz_words: List[tuple]
    waiting_for_quiz_answers: bool
    last_user_input: str
    session_active: bool
    pending_updates: Dict[str, bool]
    conversation_context: List[Dict[str, str]]
    study_words_cache: List[tuple]  # Recently studied words for notes integration
    last_action: str  # Track last node for context

class GREVocabAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
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
                "session_score": 0.0
            },
            current_quiz_words=[],
            waiting_for_quiz_answers=False,
            last_user_input="",
            session_active=False,
            pending_updates={},
            conversation_context=[],
            study_words_cache=[],
            last_action=""
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("start_quiz", self._quiz_node)
        workflow.add_node("study_words", self._study_node)
        workflow.add_node("notes", self._notes_node)
        workflow.add_node("progress", self._progress_node)
        workflow.add_node("reset_progress", self._reset_node)
        workflow.add_node("end_session", self._end_session_node)
        workflow.add_node("general", self._general_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "start_quiz": "start_quiz",
                "study_words": "study_words",
                "notes": "notes",
                "progress": "progress",
                "reset_progress": "reset_progress",
                "end_session": "end_session",
                "general": "general",
                "end": END
            }
        )

        # All nodes end the workflow
        for node in ["start_quiz", "study_words", "notes", "progress", 
                    "reset_progress", "end_session", "general"]:
            workflow.add_edge(node, END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        user_input = state["last_user_input"]

        # Handle quiz answers first
        if state["waiting_for_quiz_answers"]:
            state["current_mode"] = "start_quiz"
            return state

        # Check for natural language notes commands first
        notes_action = self._check_notes_commands(user_input, state)
        if notes_action:
            return self._handle_notes_command(state, notes_action)

        # Classify intent using merged LLM classification
        intent = self._classify_intent(user_input, state)
        state["current_mode"] = intent
        
        return state
    
    def _check_notes_commands(self, user_input: str, state: AgentState) -> Dict:
        """Check for natural language notes commands like 'move X to notes'"""
        
        # Build context from study session
        study_context = ""
        if state["study_words_cache"]:
            studied_words = [w[0] for w in state["study_words_cache"]]
            study_context = f"Recently studied words: {studied_words}"
        
        prompt = f"""
        Check if this is a notes-related command for a vocabulary app.
        
        Context: {study_context}
        User input: "{user_input}"
        
        Detect these patterns:
        1. "move [word] to notes" - move specific word from study to notes
        2. "move all to notes" - move all studied words to notes
        3. "move [word1] and [word2] to notes" - move multiple specific words
        4. Questions about notes like "what notes do I have about memory?"
        5. Regular note commands like "note for [word]: [content]"
        
        If it's a notes command, respond in JSON:
        {{
            "is_notes_command": true,
            "action": "move_to_notes|query_notes|add_note",
            "words": ["word1", "word2"] or null,
            "query": "search query" or null,
            "note_content": "content" or null
        }}
        
        If not a notes command, respond:
        {{"is_notes_command": false}}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.strip())
            return result if result.get("is_notes_command") else None
        except Exception as e:
            print(f"Notes command detection error: {e}")
            # Fallback: simple pattern matching
            user_input_lower = user_input.lower()
            if "move" in user_input_lower and "notes" in user_input_lower:
                if "all" in user_input_lower:
                    return {"is_notes_command": True, "action": "move_to_notes", "words": ["all"]}
                else:
                    # Try to extract word names
                    words = []
                    if study_context:
                        for word, _ in state.get("study_words_cache", []):
                            if word.lower() in user_input_lower:
                                words.append(word)
                    return {"is_notes_command": True, "action": "move_to_notes", "words": words}
            elif "note for" in user_input_lower:
                return {"is_notes_command": True, "action": "add_note"}
            return None
    
    def _handle_notes_command(self, state: AgentState, notes_action: Dict) -> AgentState:
        """Handle natural language notes commands"""
        action = notes_action.get("action")
        
        if action == "move_to_notes":
            return self._move_study_to_notes(state, notes_action)
        elif action == "query_notes":
            return self._query_notes_document(state, notes_action)
        elif action == "add_note":
            # Process the note directly instead of just setting mode
            state["current_mode"] = "notes"
            return self._notes_node(state)
        else:
            state["current_mode"] = "notes"
            return state
    
    def _move_study_to_notes(self, state: AgentState, notes_action: Dict) -> AgentState:
        """Move study content to notes"""
        if not state["study_words_cache"]:
            response = "No recent study session to move to notes. Try studying first!"
            state["messages"].append({"role": "assistant", "content": response})
            return state
        
        words_to_move = notes_action.get("words", [])
        
        # If no specific words mentioned, move all
        if not words_to_move or "all" in str(words_to_move).lower():
            words_to_move = [w[0] for w in state["study_words_cache"]]
        
        moved_count = 0
        response = "📝 **Moved to Notes:**\n\n"
        
        for word_name in words_to_move:
            # Find the word in study cache
            study_word = None
            for word, definition in state["study_words_cache"]:
                if word.lower() == word_name.lower():
                    study_word = (word, definition)
                    break
            
            if study_word:
                word, definition = study_word
                
                # Create comprehensive note from study content
                note_content = f"Definition: {definition}"
                
                # Add example if available
                example = get_example_sentence(word)
                if example:
                    note_content += f" | Example: {example}"
                
                # Save to database and RAG
                save_note(word, note_content, "study_moved")
                if self.rag_system:
                    try:
                        self.rag_system.add_note_to_rag(word, note_content, "study_moved")
                    except:
                        pass
                
                response += f"• **{word.upper()}**: {note_content}\n"
                moved_count += 1
        
        if moved_count > 0:
            response += f"\n✅ **Moved {moved_count} word(s) to notes!**\n"
        else:
            response = "❌ Couldn't find those words in your recent study session."
        
        state["messages"].append({"role": "assistant", "content": response})
        state["current_mode"] = "notes"  # Set state to notes
        state["last_action"] = "notes"
        return state
    
    def _query_notes_document(self, state: AgentState, notes_action: Dict) -> AgentState:
        """Query notes like a document using RAG"""
        query = notes_action.get("query", "")
        
        if not query:
            # Show all notes if no specific query
            state["current_mode"] = "notes"
            return state
        
        if not self.rag_system:
            response = "RAG system not available for note search."
            state["messages"].append({"role": "assistant", "content": response})
            return state
        
        try:
            # Use RAG to search notes semantically
            similar_notes = self.rag_system.search_similar_notes(query, 10)
            
            if similar_notes:
                response = f"🔍 **Notes about '{query}':**\n\n"
                
                for note in similar_notes:
                    similarity_pct = note["similarity"] * 100
                    response += f"**{note['word'].upper()}** ({similarity_pct:.0f}% match)\n"
                    response += f"• {note['note']}\n\n"
                
                response += "💡 **Ask me anything about your notes!** Like:\n"
                response += "• 'Which words are about memory techniques?'\n"
                response += "• 'What have I learned about difficult words?'\n"
                response += "• 'Show me notes with examples'\n"
            else:
                response = f"No notes found matching '{query}'. Try different search terms or add more notes first."
        except Exception as e:
            response = f"Error searching notes: {e}"
        
        state["messages"].append({"role": "assistant", "content": response})
        state["last_action"] = "notes"
        return state
    
    def _classify_intent(self, user_input: str, state: AgentState) -> str:
        """Merged intent classification using LLM"""
        
        # Build context from recent conversation and last action
        context_text = ""
        if state["conversation_context"]:
            recent_messages = state["conversation_context"][-2:]
            context_text = "\n".join([
                f"{msg['role']}: {msg['content'][:100]}..." 
                for msg in recent_messages
            ])
        
        if state["last_action"]:
            context_text += f"\nLast action: {state['last_action']}"
        
        if state["study_words_cache"]:
            context_text += f"\nRecently studied words: {[w[0] for w in state['study_words_cache'][:3]]}"

        prompt = f"""
        Classify the user's intent for a GRE vocabulary learning app.
        
        Available intents:
        - start_quiz: wants to take a quiz or practice (e.g., "quiz", "test me", "quiz 5 words")
        - study_words: wants to study/review words (e.g., "study", "review words", "show difficult words")
        - notes: wants to add, view, or manage notes (e.g., "add note", "view notes", "note for sparse")
        - progress: wants to see progress/statistics (e.g., "dashboard", "my progress", "how am I doing")
        - reset_progress: wants to reset all progress (e.g., "reset", "start over", "clear progress")
        - end_session: wants to end current session (e.g., "end session", "finish", "save progress")
        - general: general conversation or unclear intent
        
        Context: {context_text}
        User input: "{user_input}"
        
        Respond with only the intent name.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            intent = response.content.strip().lower()
            
            valid_intents = ["start_quiz", "study_words", "notes", "progress", "reset_progress", "end_session", "general"]
            return intent if intent in valid_intents else "general"
        except:
            return "general"

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
        quiz_params = self._extract_quiz_parameters(user_input)

        # Start session if not active
        if not state["session_active"]:
            state["session_active"] = True
            state["pending_updates"] = {}

        # Get quiz words using database sampling
        quiz_words = sample_words_for_quiz(quiz_params["word_count"])

        if not quiz_words:
            response = "Sorry, couldn't generate quiz. Database might be empty."
            state["messages"].append({"role": "assistant", "content": response})
            return state

        state["current_quiz_words"] = quiz_words
        state["waiting_for_quiz_answers"] = True
        state["last_action"] = "quiz"

        # Create quiz with better formatting
        response = f"🧠 **Quiz Time!** ({len(quiz_words)} words)\n\n"
        for i, (word, _) in enumerate(quiz_words, 1):
            response += f"**{i}. {word.upper()}**\n"
        
        response += f"\n💡 **Instructions:** Provide definitions for each word (number them or just list them)\n"

        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _extract_quiz_parameters(self, user_input: str) -> Dict:
        """Extract quiz parameters from user input using simple regex"""
        import re
        
        # Look for numbers in the input
        numbers = re.findall(r'\b(\d+)\b', user_input.lower())
        
        if numbers:
            # Take the first number found
            count = int(numbers[0])
            # Clamp between 1 and 10
            count = min(max(count, 1), 10)
            return {"word_count": count}
        else:
            # Default to 5 if no number found
            return {"word_count": 5}
    


    def _process_quiz_answers(self, state: AgentState) -> AgentState:
        user_answers = state["last_user_input"]
        quiz_words = state["current_quiz_words"]

        # Grade answers using LLM
        results = self._grade_answers(quiz_words, user_answers)

        # Create simplified feedback summary
        feedback = "📊 **Quiz Results:**\n\n"
        
        correct_count = 0
        for word, correct_def in quiz_words:
            is_correct = results.get(word, False)
            if is_correct:
                correct_count += 1
            
            status = "✅" if is_correct else "❌"
            feedback += f"{status} **{word.upper()}**: {correct_def}\n"
            
            # Add example sentence
            example = get_example_sentence(word)
            if not example:
                example = self._generate_example_sentence(word, correct_def)
                if example:
                    save_example_sentence(word, example)
            
            if example:
                feedback += f"   *Example: {example}*\n"
            feedback += "\n"

        # Update performance tracking
        total_count = len(results)
        session_score = correct_count / total_count if total_count > 0 else 0

        state["session_quiz_performance"]["words_attempted"].extend([w for w, _ in quiz_words])
        state["session_quiz_performance"]["session_score"] = session_score

        # Update word statuses immediately
        for word, correct in results.items():
            update_word_status(word, correct)

        # Performance summary
        feedback += f"📈 **Score: {session_score:.1%}** ({correct_count}/{total_count})\n\n"
        
        if session_score >= 0.8:
            feedback += "🎉 Excellent work!\n"
        elif session_score >= 0.6:
            feedback += "👍 Good progress!\n"
        else:
            feedback += "💪 Keep practicing!\n"

        # Reset quiz state
        state["current_quiz_words"] = []
        state["waiting_for_quiz_answers"] = False

        state["messages"].append({"role": "assistant", "content": feedback})
        return state

    def _grade_answers(self, quiz_words: List[tuple], user_answers: str) -> Dict[str, bool]:
        """Grade answers using simplified LLM"""
        results = {}

        words_info = "\n".join([f"{word}: {definition}" for word, definition in quiz_words])

        prompt = f"""
        Grade these quiz answers fairly. Accept correct definitions, close synonyms, and reasonable interpretations.
        
        Words and correct definitions:
        {words_info}
        
        User's answers: "{user_answers}"
        
        Mark as CORRECT if the user:
        - Gives the correct definition or close synonym
        - Shows clear understanding of the main meaning
        - Uses related words that capture the essence
        
        Mark as INCORRECT if:
        - Definition is wrong or unrelated
        - Shows no understanding of the word
        - Gives opposite meaning
        - No answer provided for that word
        
        Be fair but not overly generous.
        
        Respond in JSON format:
        {{"word1": true/false, "word2": true/false, ...}}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            grading_result = json.loads(response.content.strip())
            
            for word, _ in quiz_words:
                results[word] = grading_result.get(word.lower(), False)
        except:
            # Fallback: grade each word individually
            for word, correct_def in quiz_words:
                results[word] = self._grade_single_word(word, correct_def, user_answers)

        return results

    def _grade_single_word(self, word: str, correct_def: str, user_answers: str) -> bool:
        """Grade a single word with fair scoring"""
        prompt = f"""
        Does the user's answer show correct understanding of "{word}" (definition: "{correct_def}")?
        
        User's answers: "{user_answers}"
        
        Accept correct definitions, close synonyms, and reasonable interpretations.
        Reject wrong definitions, unrelated answers, or no attempt.
        
        Answer only "YES" or "NO".
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return "yes" in response.content.lower()
        except:
            return False

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
            "session_type": "mixed"
        }
        
        try:
            save_session_data(session_data)
        except Exception as e:
            print(f"Error saving session: {e}")

        # Generate session summary
        perf = state["session_quiz_performance"]
        total_attempted = len(perf["words_attempted"])
        session_score = perf.get("session_score", 0)

        response = "🎯 **Session Complete!**\n\n"
        response += f"📊 **Final Stats:**\n"
        response += f"• Words Attempted: {total_attempted}\n"
        response += f"• Session Score: {session_score:.1%}\n"
        response += f"• Word Statuses Updated: {updated_count}\n\n"

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
        """Simple progress dashboard with red/green dots"""
        from database import get_recent_quiz_results
        
        status_counts = get_word_status_counts()
        total_words = sum(status_counts.values())
        
        if total_words == 0:
            response = "📊 **Dashboard**: No vocabulary data yet. Start with a quiz!"
            state["messages"].append({"role": "assistant", "content": response})
            return state
        
        response = "📊 **Your Progress**\n\n"
        response += f"**Total Words:** {total_words}\n"
        response += f"**Strong:** {status_counts['strong']} | **Moderate:** {status_counts['moderate']} | **Weak:** {status_counts['weak']} | **Unknown:** {status_counts['unknown']}\n\n"
        
        # Recent quiz performance (last 10 words) - red/green dots only
        recent_results = get_recent_quiz_results(10)
        if recent_results:
            response += "**Recent Quiz Performance:**\n"
            dots = ""
            for word, correct in recent_results:
                dots += "🟢" if correct else "🔴"
            response += f"{dots}\n"
            response += f"*Last {len(recent_results)} words*\n\n"
        
        # Session stats
        if state["session_active"] and state["session_quiz_performance"]["words_attempted"]:
            perf = state["session_quiz_performance"]
            response += f"**Current Session:** {len(perf['words_attempted'])} words, {perf.get('session_score', 0):.1%} score\n"
        
        state["messages"].append({"role": "assistant", "content": response})
        state["last_action"] = "progress"
        return state
    
    def _reset_node(self, state: AgentState) -> AgentState:
        """Reset all progress"""
        try:
            import sqlite3
            conn = sqlite3.connect("vocab.db")
            c = conn.cursor()
            
            # Reset all word statuses to unknown
            c.execute("UPDATE vocabulary SET status = 'unknown', attempts = 0, correct_attempts = 0")
            
            # Clear notes
            c.execute("DELETE FROM user_notes")
            
            # Clear session history
            c.execute("DELETE FROM session_history")
            
            conn.commit()
            conn.close()
            
            # Reset session state
            state["session_quiz_performance"] = {
                "words_attempted": [],
                "session_score": 0.0
            }
            state["session_active"] = False
            state["study_words_cache"] = []
            
            response = "🔄 **Progress Reset Complete!**\n\n"
            response += "• All words marked as unknown\n"
            response += "• Notes cleared\n"
            response += "• Session history cleared\n\n"
            response += "Ready to start fresh! Try taking a quiz."
            
        except Exception as e:
            response = f"❌ Error resetting progress: {e}"

        state["messages"].append({"role": "assistant", "content": response})
        state["last_action"] = "reset"
        return state



    def _study_node(self, state: AgentState) -> AgentState:
        # Show weak/unknown words for study
        study_words = []
        
        # Get weak and unknown words
        weak_words = get_words_by_status("weak", 2)
        unknown_words = get_words_by_status("unknown", 3)
        
        study_words = weak_words + unknown_words
        random.shuffle(study_words)
        study_words = study_words[:5]  # Limit to 5 words
        
        if not study_words:
            response = "🎉 Great! No words to study. Try taking a quiz!"
            state["messages"].append({"role": "assistant", "content": response})
            return state

        # Cache studied words for notes integration
        state["study_words_cache"] = study_words
        state["last_action"] = "study"

        response = f"📚 **Study Session** ({len(study_words)} words):\n\n"
        
        for i, (word, definition) in enumerate(study_words, 1):
            response += f"**{i}. {word.upper()}**\n"
            response += f"📖 **Meaning:** {definition}\n"
            
            # Add example sentence
            example = get_example_sentence(word)
            if not example:
                example = self._generate_example_sentence(word, definition)
                if example:
                    save_example_sentence(word, example)
            
            if example:
                response += f"📝 **Example:** {example}\n"
            
            # Generate mnemonic/memory trick
            mnemonic = self._generate_memory_trick(word, definition)
            if mnemonic:
                response += f"🧠 **Memory Trick:** {mnemonic}\n"
            
            # Add user notes if any
            notes = get_notes_for_word(word)
            if notes:
                response += f"💭 **Your Note:** {notes[0][0]}\n"
            
            response += "\n"

        response += "💡 **Next Steps:**\n"
        response += "• Say 'move [word] to notes' to save study content\n"
        response += "• Say 'move all to notes' to save everything\n"
        response += "• Add custom notes: 'note for [word]: [content]'\n"
        response += "• Take a 'quiz' to test yourself\n"
        
        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _generate_memory_trick(self, word: str, definition: str) -> str:
        """Generate a memory trick/mnemonic for a word"""
        prompt = f"""
        Create a memorable trick to remember the word "{word}" (meaning: {definition}).
        
        Make it:
        - Short and catchy (1-2 sentences max)
        - Use wordplay, sound associations, or visual imagery
        - Connect the word's sound/spelling to its meaning
        - Be creative and fun
        
        Examples:
        - "SPARSE sounds like 'SPACE' - imagine things spread out in space"
        - "ABATE sounds like 'A BAIT' - the storm took the bait and calmed down"
        - "VERBOSE = VERB + OSE - someone who uses too many verbs!"
        
        Just give the memory trick, no extra text.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except:
            return ""

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
        
        # Enhanced notes management with study integration and RAG
        return self._handle_notes_with_rag(state)
    
    def _handle_notes_with_rag(self, state: AgentState) -> AgentState:
        """Enhanced notes management with RAG and study integration"""
        user_input = state["last_user_input"]
        
        # Use LLM to determine note action and extract info
        note_action = self._analyze_note_request(user_input, state)
        
        if note_action["action"] == "add":
            return self._add_note_with_rag(state, note_action)
        elif note_action["action"] == "view":
            return self._view_notes_with_rag(state, note_action)
        elif note_action["action"] == "search":
            return self._search_notes_with_rag(state, note_action)
        else:
            return self._notes_help(state)
    
    def _analyze_note_request(self, user_input: str, state: AgentState) -> Dict:
        """Analyze what the user wants to do with notes"""
        
        # Build context from study session
        context = ""
        if state["study_words_cache"]:
            studied_words = [w[0] for w in state["study_words_cache"]]
            context = f"Recently studied: {', '.join(studied_words)}"
        
        prompt = f"""Parse this note command. Be flexible with formats.

User: "{user_input}"
{context}

Actions:
- add: Adding a note (note for X, X means Y, remember X)
- view: Viewing notes (show notes, view notes, my notes)  
- search: Searching notes (find notes about X)

For ADD: Extract word and note content from ANY format.
For SEARCH: Extract the search topic.

Respond ONLY in this JSON format:
{{"action": "add/view/search", "word": "word_name", "note_content": "content", "search_query": "query"}}

Examples:
"note for sparse: thin" -> {{"action": "add", "word": "sparse", "note_content": "thin", "search_query": null}}
"show my notes" -> {{"action": "view", "word": null, "note_content": null, "search_query": null}}"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            result = json.loads(response.content.strip())
            # Ensure all required keys exist
            return {
                "action": result.get("action", "view"),
                "word": result.get("word"),
                "note_content": result.get("note_content"), 
                "search_query": result.get("search_query")
            }
        except:
            return {"action": "view", "word": None, "note_content": None, "search_query": None}
    
    def _add_note_with_rag(self, state: AgentState, note_info: Dict) -> AgentState:
        """Add note with RAG enhancement"""
        word = note_info.get("word")
        note_content = note_info.get("note_content")
        
        if not word or not note_content:
            response = "❌ **Couldn't parse your note.**\n\n"
            response += "**Try:** 'note for [word]: [content]'\n"
            response += "**Example:** 'note for sparse: means thin and scattered'\n"
            
            if state["study_words_cache"]:
                words = [w[0] for w in state["study_words_cache"][:3]]
                response += f"\n**Recently studied:** {', '.join(words)}"
            
            state["messages"].append({"role": "assistant", "content": response})
            return state
        
        # Save note
        save_note(word, note_content, "user")
        if self.rag_system:
            self.rag_system.add_note_to_rag(word, note_content, "user")
        
        # Get definition
        definition = get_word_definition(word)
        
        response = f"✅ **Note saved for '{word.upper()}'**\n\n"
        response += f"📝 *Your note:* {note_content}\n"
        if definition:
            response += f"📖 *Definition:* {definition}\n"
        
        state["messages"].append({"role": "assistant", "content": response})
        state["current_mode"] = "notes"  # Set state to notes
        state["last_action"] = "notes"
        return state
    
    def _view_notes_with_rag(self, state: AgentState, note_info: Dict) -> AgentState:
        """View notes with RAG organization"""
        word = note_info.get("word")
        
        if word:
            # Show notes for specific word
            notes = get_notes_for_word(word)
            if notes:
                response = f"📝 **Notes for {word.upper()}:**\n\n"
                for note_content, note_type, timestamp in notes:
                    response += f"• {note_content}\n"
                
                # Show definition
                definition = get_word_definition(word)
                if definition:
                    response += f"\n*Definition:* {definition}\n"
            else:
                response = f"No notes found for **{word}**."
        else:
            # Show all notes organized by RAG
            all_notes = get_all_notes(15)
            if all_notes:
                response = "📝 **Your Vocabulary Notes:**\n\n"
                for word, note, note_type, timestamp in all_notes:
                    response += f"**{word.upper()}**: {note}\n"
                
                # Use RAG to suggest note themes
                if self.rag_system and len(all_notes) > 3:
                    response += f"\n💡 **Tip:** Try 'find notes about [memory tricks/definitions/etc]' to search your notes!\n"
            else:
                response = "You haven't created any notes yet. Try 'note for [word]: [content]'"
        
        state["messages"].append({"role": "assistant", "content": response})
        state["last_action"] = "notes"
        return state
    
    def _search_notes_with_rag(self, state: AgentState, note_info: Dict) -> AgentState:
        """Search notes using RAG semantic search"""
        search_query = note_info.get("search_query", "")
        
        if not search_query or not self.rag_system:
            return self._view_notes_with_rag(state, {"word": None})
        
        try:
            # Use RAG to find semantically similar notes
            similar_notes = self.rag_system.search_similar_notes(search_query, 8)
            
            if similar_notes:
                response = f"🔍 **Notes matching '{search_query}':**\n\n"
                for note in similar_notes:
                    similarity_pct = note["similarity"] * 100
                    response += f"**{note['word'].upper()}** ({similarity_pct:.0f}% match)\n"
                    response += f"• {note['note']}\n\n"
            else:
                response = f"No notes found matching '{search_query}'. Try different search terms."
        except:
            response = "Search temporarily unavailable. Showing all notes instead."
            return self._view_notes_with_rag(state, {"word": None})
        
        state["messages"].append({"role": "assistant", "content": response})
        state["last_action"] = "notes"
        return state
    
    def _notes_help(self, state: AgentState) -> AgentState:
        """Show notes help with study integration"""
        response = "📝 **Notes Help:**\n\n"
        
        # Show study context if available
        if state["last_action"] == "study" and state["study_words_cache"]:
            studied_words = [w[0] for w in state["study_words_cache"][:3]]
            response += f"**Recently studied:** {', '.join(studied_words)}\n\n"
        
        response += "**Commands:**\n"
        response += "• `note for [word]: [content]` - Add a note\n"
        response += "• `view notes` - See all your notes\n"
        response += "• `find notes about [topic]` - Search notes with RAG\n\n"
        
        response += "**Examples:**\n"
        response += "• `note for sparse: means thin and scattered`\n"
        response += "• `find notes about memory tricks`\n"
        
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