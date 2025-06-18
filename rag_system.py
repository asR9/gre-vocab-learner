import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings
from database import get_all_notes, get_word_definition, save_note
import json

class RAGSystem:
    def __init__(self):
        """Initialize ChromaDB and embeddings"""
        self.embeddings_available = False
        
        try:
            # Initialize OpenAI embeddings without testing
            self.embeddings = OpenAIEmbeddings()
            self.embeddings_available = True
            print("✓ OpenAI embeddings initialized")
        except Exception as e:
            print(f"⚠️ OpenAI embeddings not available: {e}")
            self.embeddings_available = False
            self.embeddings = None
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collections
        try:
            self.notes_collection = self.client.get_collection("vocab_notes")
        except:
            self.notes_collection = self.client.create_collection(
                name="vocab_notes",
                metadata={"description": "User notes and mnemonics for vocabulary words"}
            )
        
        try:
            self.definitions_collection = self.client.get_collection("vocab_definitions")
        except:
            self.definitions_collection = self.client.create_collection(
                name="vocab_definitions",
                metadata={"description": "Word definitions for semantic search"}
            )
    
    def add_note_to_rag(self, word: str, note_content: str, note_type: str = 'user'):
        """Add a note to the RAG system"""
        if not self.embeddings_available:
            return False
            
        try:
            # Create embedding for the note
            embedding = self.embeddings.embed_query(f"{word}: {note_content}")
            
            # Add to ChromaDB
            self.notes_collection.add(
                embeddings=[embedding],
                documents=[note_content],
                metadatas=[{
                    "word": word,
                    "note_type": note_type,
                    "content": note_content
                }],
                ids=[f"{word}_{note_type}_{len(note_content)}"]
            )
            
            return True
        except Exception as e:
            print(f"Error adding note to RAG: {e}")
            return False
    
    def add_definition_to_rag(self, word: str, definition: str):
        """Add a word definition to the RAG system"""
        if not self.embeddings_available:
            return False
            
        try:
            # Create embedding for the definition
            embedding = self.embeddings.embed_query(f"{word}: {definition}")
            
            # Add to ChromaDB
            self.definitions_collection.add(
                embeddings=[embedding],
                documents=[definition],
                metadatas=[{
                    "word": word,
                    "definition": definition
                }],
                ids=[f"def_{word}"]
            )
            
            return True
        except Exception as e:
            print(f"Error adding definition to RAG: {e}")
            return False
    
    def search_similar_notes(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar notes based on query"""
        try:
            # Create embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in notes collection
            results = self.notes_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "word": results['metadatas'][0][i]['word'],
                        "note": doc,
                        "note_type": results['metadatas'][0][i]['note_type'],
                        "similarity": 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching notes: {e}")
            return []
    
    def search_similar_words(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for words with similar definitions"""
        try:
            # Create embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in definitions collection
            results = self.definitions_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "word": results['metadatas'][0][i]['word'],
                        "definition": doc,
                        "similarity": 1 - results['distances'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching definitions: {e}")
            return []
    
    def summarize_notes(self, query: str = None) -> str:
        """Summarize user notes, optionally filtered by query"""
        try:
            if query:
                # Search for relevant notes
                relevant_notes = self.search_similar_notes(query, n_results=10)
                notes_text = "\n".join([f"- {note['word']}: {note['note']}" for note in relevant_notes])
            else:
                # Get all notes from database
                all_notes = get_all_notes(limit=20)
                notes_text = "\n".join([f"- {word}: {note}" for word, note, _, _ in all_notes])
            
            if not notes_text.strip():
                return "No notes found to summarize."
            
            return f"Summary of your vocabulary notes:\n\n{notes_text}"
        except Exception as e:
            print(f"Error summarizing notes: {e}")
            return "Error generating summary."
    
    def initialize_with_existing_data(self, max_items=10):
        """Initialize RAG system with existing notes and definitions (limited for demo)"""
        try:
            # Add existing notes (limited)
            notes = get_all_notes(limit=max_items)
            notes_added = 0
            for word, note_content, note_type, _ in notes:
                if self.add_note_to_rag(word, note_content, note_type):
                    notes_added += 1
            
            # Add existing definitions (very limited sample)
            from database import sample_words_for_quiz
            words = sample_words_for_quiz(max_items)  # Small sample
            definitions_added = 0
            for word, definition in words:
                if self.add_definition_to_rag(word, definition):
                    definitions_added += 1
            
            print(f"Initialized RAG system with {notes_added} notes and {definitions_added} definitions")
        except Exception as e:
            print(f"Warning: Could not fully initialize RAG system: {e}")
            print("RAG features will be limited until embeddings are available")
    
    def find_related_words(self, word: str, n_results: int = 5) -> List[Dict]:
        """Find words related to the given word"""
        definition = get_word_definition(word)
        if not definition:
            return []
        
        # Search for similar definitions
        similar_words = self.search_similar_words(definition, n_results + 1)  # +1 to exclude self
        
        # Filter out the original word
        related_words = [w for w in similar_words if w['word'].lower() != word.lower()]
        
        return related_words[:n_results]