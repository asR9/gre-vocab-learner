from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import random
from database import (
    sample_words_for_quiz,
    update_word_status,
    get_words_by_status,
    save_session_data,
    get_word_definition,
    batch_update_word_statuses
)

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_mode: str
    session_quiz_performance: Dict[str, Any]
    generated_mnemonics: Dict[str, str]
    user_notes: Dict[str, str]
    current_quiz_words: List[tuple]
    waiting_for_quiz_answers: bool
    last_user_input: str

class GREVocabAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

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
            last_user_input=""
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("quiz", self._quiz_node)
        workflow.add_node("study", self._study_node)
        workflow.add_node("notes", self._notes_node)
        workflow.add_node("general", self._general_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "quiz": "quiz",
                "study": "study",
                "notes": "notes",
                "general": "general",
                "end": END
            }
        )

        workflow.add_edge("quiz", END)
        workflow.add_edge("study", END)
        workflow.add_edge("notes", END)
        workflow.add_edge("general", END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        user_input = state["last_user_input"].lower()

        # If waiting for quiz answers, stay in quiz mode
        if state["waiting_for_quiz_answers"]:
            state["current_mode"] = "quiz"
            return state

        # Simple keyword routing
        if any(word in user_input for word in ["quiz", "test", "practice"]):
            state["current_mode"] = "quiz"
        elif any(word in user_input for word in ["study", "learn", "mnemonic", "help"]):
            state["current_mode"] = "study"
        elif any(word in user_input for word in ["note", "write", "update", "show"]):
            state["current_mode"] = "notes"
        else:
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
        user_input = state["last_user_input"].lower()

        # Check if user requested specific words
        from database import search_words
        requested_words = []
        input_words = user_input.split()

        for input_word in input_words:
            clean_word = ''.join(c for c in input_word if c.isalpha())
            if len(clean_word) > 3:  # Only check words longer than 3 chars
                matches = search_words(clean_word, 5)
                for word, definition in matches:
                    if word.lower() == clean_word:
                        requested_words.append((word, definition))
                        break

        if requested_words:
            quiz_words = requested_words
        else:
            # Extract number from input, default to 5
            quiz_count = 5
            for word in input_words:
                if word.isdigit():
                    quiz_count = min(int(word), 10)
                    break

            # Get random quiz words
            quiz_words = sample_words_for_quiz(quiz_count)

        if not quiz_words:
            response = "Sorry, couldn't generate quiz. Database might be empty."
            state["messages"].append({"role": "assistant", "content": response})
            return state

        state["current_quiz_words"] = quiz_words
        state["waiting_for_quiz_answers"] = True

        # Create quiz
        response = f"🧠 **Quiz Time!** Define these {len(quiz_words)} words:\n\n"
        for i, (word, _) in enumerate(quiz_words, 1):
            response += f"{i}. **{word}**\n"
        response += "\n💡 *Just provide your definitions (you can separate them with numbers or just list them)*"

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _process_quiz_answers(self, state: AgentState) -> AgentState:
        user_answers = state["last_user_input"]
        quiz_words = state["current_quiz_words"]

        # Grade answers using simple keyword matching
        results = self._grade_answers(quiz_words, user_answers)

        # Create feedback
        feedback = "📊 **Quiz Results:**\n\n"
        for word, correct_def in quiz_words:
            status = "✅ Correct" if results.get(word, False) else "❌ Incorrect"
            feedback += f"**{word}**: {status}\n"
            feedback += f"*Definition:* {correct_def}\n\n"

        # Update performance
        correct_count = sum(results.values())
        total_count = len(results)
        session_score = correct_count / total_count if total_count > 0 else 0

        state["session_quiz_performance"]["words_attempted"].extend([w for w, _ in quiz_words])
        state["session_quiz_performance"]["correct_answers"].extend([w for w, correct in results.items() if correct])
        state["session_quiz_performance"]["incorrect_answers"].extend([w for w, correct in results.items() if not correct])
        state["session_quiz_performance"]["weak_words"].extend([w for w, correct in results.items() if not correct])
        state["session_quiz_performance"]["session_score"] = session_score

        # Update database
        batch_update_word_statuses(results)

        feedback += f"📈 **Score: {session_score:.1%}** ({correct_count}/{total_count})\n\n"
        feedback += "What would you like to do next? (quiz, study, or notes)"

        # Reset quiz state
        state["current_quiz_words"] = []
        state["waiting_for_quiz_answers"] = False

        state["messages"].append({"role": "assistant", "content": feedback})
        return state

    def _grade_answers(self, quiz_words: List[tuple], user_answers: str) -> Dict[str, bool]:
        """Grade answers using LLM only"""
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



    def _study_node(self, state: AgentState) -> AgentState:
        # Get weak words from session or database
        weak_words = state["session_quiz_performance"]["weak_words"]
        if not weak_words:
            weak_from_db = get_words_by_status("weak", 3) + get_words_by_status("unknown", 3)
            weak_words = [word for word, _ in weak_from_db[:5]]

        if not weak_words:
            response = "Great! No weak words to study. Try taking a quiz first!"
            state["messages"].append({"role": "assistant", "content": response})
            return state

        # Generate ALL mnemonics in ONE fast LLM call
        study_words = weak_words[:3]
        word_definitions = [(word, get_word_definition(word)) for word in study_words]
        mnemonics = self._generate_batch_mnemonics(word_definitions)

        response = "📖 **Study Session - Focus Words:**\n\n"
        for i, (word, definition) in enumerate(word_definitions):
            mnemonic = mnemonics[i] if i < len(mnemonics) else f"Remember: {word.upper()} = {definition}"
            state["generated_mnemonics"][word] = mnemonic

            response += f"**{word.upper()}**\n"
            response += f"*Definition:* {definition}\n"
            response += f"*Memory Trick:* {mnemonic}\n\n"

        response += "💡 *Ready for another quiz or want to check your notes?*"
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
        user_input_lower = user_input.lower()

        # Check if user wants to add a note with format: "note for word: content"
        if "note for" in user_input_lower and ":" in user_input:
            try:
                # Parse "note for [word]: [content]"
                parts = user_input.split(":")
                word_part = parts[0].lower().replace("note for", "").strip()
                note_content = ":".join(parts[1:]).strip()

                # Save the note
                state["user_notes"][word_part] = note_content

                response = f"✅ **Note saved for '{word_part}':**\n{note_content}\n\nWhat would you like to do next?"
                state["messages"].append({"role": "assistant", "content": response})
                return state
            except:
                pass

        # Show current notes
        all_notes = {}
        all_notes.update(state["generated_mnemonics"])
        all_notes.update(state["user_notes"])

        if not all_notes:
            response = """📝 **Your Notes are Empty**

Study some words first to generate notes, or add your own by saying:
**"note for [word]: [your note]"**

Example: "note for sparse: remember it means thin or scanty"

What would you like to do? (quiz, study, or add notes)"""
        else:
            response = "📝 **Your Vocabulary Notes:**\n\n"
            for word, note in list(all_notes.items())[:10]:  # Show first 10
                definition = get_word_definition(word)
                response += f"**{word.upper()}**\n"
                if definition:
                    response += f"*Definition:* {definition}\n"
                response += f"*Notes:* {note}\n\n"

            response += """
💡 **To add/edit notes, say:**
"note for [word]: [your note]"

Example: "note for abate: think of a bat flying away"

What would you like to do next? (quiz, study, or edit notes)"""

        state["messages"].append({"role": "assistant", "content": response})
        return state

    def _general_node(self, state: AgentState) -> AgentState:
        response = """
I'm your GRE Vocab Tutor! Here's what I can do:

🧠 **Quiz**: "Start a quiz" or "Quiz me on 5 words"
📖 **Study**: "Help me study" or "Show me mnemonics"
📝 **Notes**: "Show my notes" or "note for [word]: [your note]"

What would you like to do?
        """
        state["messages"].append({"role": "assistant", "content": response})
        return state

    def process_message(self, user_input: str) -> str:
        """Process user message and return response"""
        self.state["last_user_input"] = user_input

        try:
            result = self.graph.invoke(self.state)
            self.state = result

            if self.state["messages"]:
                return self.state["messages"][-1]["content"]
            else:
                return "I'm not sure how to help. Try asking about quiz, study, or notes!"
        except Exception as e:
            return f"Sorry, error occurred: {str(e)}"

    def save_session_progress(self):
        """Save session progress"""
        session_data = {
            "session_quiz_performance": self.state["session_quiz_performance"],
            "generated_mnemonics": self.state["generated_mnemonics"],
            "user_notes": self.state["user_notes"]
        }
        save_session_data(session_data)