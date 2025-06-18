from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import Dict, List
import json

class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Define available intents
        self.intents = {
            "start_quiz": "User wants to start a quiz or practice session",
            "study_words": "User wants to study words, get mnemonics, or learn",
            "add_note": "User wants to add or update a note for a word",
            "view_notes": "User wants to see their notes or review them",
            "summarize_notes": "User wants a summary of their notes",
            "get_hints": "User wants hints during a quiz",
            "end_session": "User wants to end the current session",
            "search_words": "User wants to find or search for specific words",
            "get_progress": "User wants to see their progress or statistics",
            "general": "General conversation or unclear intent"
        }
    
    def classify_intent(self, user_input: str, conversation_context: List[Dict] = None) -> Dict:
        """Classify user intent using LLM"""
        
        # Build context from recent conversation
        context_text = ""
        if conversation_context:
            recent_messages = conversation_context[-3:]  # Last 3 messages
            context_text = "\n".join([
                f"{msg['role']}: {msg['content'][:100]}..." 
                for msg in recent_messages
            ])
        
        # Create the classification prompt
        prompt = f"""
        You are an intent classifier for a GRE vocabulary learning app. 
        
        Available intents and their descriptions:
        {json.dumps(self.intents, indent=2)}
        
        Recent conversation context:
        {context_text}
        
        Current user input: "{user_input}"
        
        Classify the user's intent based on their input and context. Consider:
        1. Direct keywords and phrases
        2. Conversation flow and context
        3. Implied actions from the user's request
        
        Respond with ONLY the intent name (e.g., "start_quiz", "study_words", etc.).
        If the intent is unclear or doesn't fit any category, respond with "general".
        
        Examples:
        - "Start a quiz with 5 words" → start_quiz
        - "I want to practice" → start_quiz  
        - "Help me study difficult words" → study_words
        - "Show me mnemonics" → study_words
        - "Note for sparse: means thin" → add_note
        - "What are my notes?" → view_notes
        - "Summarize my vocabulary notes" → summarize_notes
        - "I need a hint" → get_hints
        - "End this session" → end_session
        - "Find words like 'happy'" → search_words
        - "How am I doing?" → get_progress
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            intent = response.content.strip().lower()
            
            # Validate the intent
            if intent in self.intents:
                return {
                    "intent": intent,
                    "confidence": 0.9,  # High confidence for LLM classification
                    "description": self.intents[intent]
                }
            else:
                return {
                    "intent": "general",
                    "confidence": 0.5,
                    "description": "Fallback to general conversation"
                }
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return {
                "intent": "general",
                "confidence": 0.1,
                "description": "Error in classification, defaulting to general"
            }
    
    def extract_quiz_parameters(self, user_input: str) -> Dict:
        """Extract quiz parameters from user input"""
        prompt = f"""
        Extract quiz parameters from this user input: "{user_input}"
        
        Look for:
        1. Number of words (default: 5, max: 10)
        2. Specific words mentioned
        3. Difficulty level (if mentioned)
        4. Word status preference (weak, unknown, etc.)
        
        Respond in JSON format:
        {{
            "word_count": <number>,
            "specific_words": [<list of words if any>],
            "difficulty": "<easy/medium/hard or null>",
            "focus_status": "<weak/unknown/moderate/strong or null>"
        }}
        
        Examples:
        - "Quiz me on 3 words" → {{"word_count": 3, "specific_words": [], "difficulty": null, "focus_status": null}}
        - "Test me on sparse and abate" → {{"word_count": 2, "specific_words": ["sparse", "abate"], "difficulty": null, "focus_status": null}}
        - "Quiz my weak words" → {{"word_count": 5, "specific_words": [], "difficulty": null, "focus_status": "weak"}}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return json.loads(response.content.strip())
        except Exception as e:
            print(f"Error extracting quiz parameters: {e}")
            return {
                "word_count": 5,
                "specific_words": [],
                "difficulty": None,
                "focus_status": None
            }
    
    def extract_note_info(self, user_input: str) -> Dict:
        """Extract note information from user input"""
        prompt = f"""
        Extract note information from this user input: "{user_input}"
        
        Look for patterns like:
        - "note for [word]: [content]"
        - "add note [word] [content]"
        - "remember [word] as [content]"
        
        Respond in JSON format:
        {{
            "word": "<word or null>",
            "note_content": "<note content or null>",
            "action": "<add/view/update>"
        }}
        
        Examples:
        - "note for sparse: means thin and scattered" → {{"word": "sparse", "note_content": "means thin and scattered", "action": "add"}}
        - "update note for abate" → {{"word": "abate", "note_content": null, "action": "update"}}
        - "show notes for difficult words" → {{"word": null, "note_content": null, "action": "view"}}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return json.loads(response.content.strip())
        except Exception as e:
            print(f"Error extracting note info: {e}")
            return {
                "word": None,
                "note_content": None,
                "action": "view"
            }