# 🎓 Enhanced GRE Vocab Tutor

An intelligent vocabulary learning application powered by advanced AI features including LLM-based intent classification, RAG (Retrieval-Augmented Generation), and session-based learning.

## ✨ New Features (Priority 1 & 2 Implementation)

### 🧠 **Priority 1: High Impact, Low Effort**

#### 1. **LLM Intent Classification**
- **Replaced keyword-based routing** with intelligent LLM-powered intent classification
- **Context-aware understanding** of user inputs with conversation history
- **Supported intents**: `start_quiz`, `study_words`, `add_note`, `view_notes`, `summarize_notes`, `get_hints`, `end_session`, `search_words`, `get_progress`, `general`
- **Graceful fallback** for unclear intents

#### 2. **Enhanced Quiz Feedback**
- **Detailed LLM grading** with constructive feedback for each answer
- **Automatic example sentence generation** for better context understanding
- **Progressive encouragement** based on performance levels
- **Smart status updates** with first-attempt tracking

#### 3. **Session-based Updates**
- **Batch database operations** at session end for better performance
- **Pending updates tracking** during active sessions
- **Session summaries** with detailed performance analytics
- **Clear session boundaries** with start/end management

#### 4. **Basic Progress Dashboard**
- **Real-time session metrics** in enhanced sidebar
- **Visual progress indicators** with charts and progress bars
- **Quick action buttons** for instant access to features
- **Overall mastery tracking** with word status breakdown

### 🔍 **Priority 2: Medium Impact, Medium Effort**

#### 1. **Chroma DB Integration for RAG**
- **Vector embeddings** for all notes and definitions using OpenAI embeddings
- **Semantic search capabilities** for finding similar words and concepts
- **Smart note summarization** with query-based filtering
- **Cross-reference functionality** linking related vocabulary

#### 2. **Hint System**
- **Progressive hints** during quizzes (first letter, word length, related concepts)
- **Context-aware suggestions** using RAG system
- **Non-intrusive help** that doesn't break quiz flow

#### 3. **Markdown Notes Support**
- **Rich text formatting** for better note organization
- **Persistent storage** in dedicated database table
- **Integration with RAG system** for intelligent search
- **Multiple note types** (user notes, mnemonics, hints)

#### 4. **Example Sentences**
- **Auto-generated contextual examples** for each vocabulary word
- **Persistent storage** to avoid regeneration
- **Integration with study sessions** for enhanced learning

## 🏗️ **Enhanced Architecture**

### **Database Schema Updates**
```sql
-- Enhanced vocabulary table
ALTER TABLE vocabulary ADD COLUMN first_attempt BOOLEAN DEFAULT NULL;
ALTER TABLE vocabulary ADD COLUMN example_sentence TEXT;

-- New dedicated notes table
CREATE TABLE user_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    note_content TEXT NOT NULL,
    note_type TEXT DEFAULT 'user',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced session tracking
ALTER TABLE session_history ADD COLUMN session_type TEXT DEFAULT 'mixed';
ALTER TABLE session_history ADD COLUMN notes_created INTEGER DEFAULT 0;
```

### **Multi-Agent System Enhancement**
- **Expanded node types**: 10 specialized nodes for different intents
- **Intelligent routing** based on LLM classification
- **Context preservation** across conversation turns
- **State management** for complex session flows

### **RAG System Architecture**
- **ChromaDB collections**: Separate collections for notes and definitions
- **OpenAI embeddings**: High-quality vector representations
- **Similarity search**: Semantic matching for related concepts
- **Batch operations**: Efficient data processing

## 🚀 **Getting Started**

### **Prerequisites**
```bash
pip install -r requirements.txt
```

### **Setup**
1. **Set OpenAI API Key**:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

2. **Initialize Database**:
   ```bash
   python -c "from database import init_database; init_database()"
   ```

3. **Test Installation**:
   ```bash
   python test_enhanced_features.py
   ```

### **Run Application**
```bash
streamlit run app.py --server.port 12000 --server.address 0.0.0.0
```

## 📖 **Usage Guide**

### **Smart Quiz System**
- **Adaptive selection**: Prioritizes unknown/weak words
- **Custom parameters**: "Quiz me on 5 words", "Quiz weak words"
- **Detailed feedback**: LLM-powered grading with explanations
- **Hint system**: Progressive help during quizzes
- **Session tracking**: Batch updates at session end

### **Enhanced Study Mode**
- **Comprehensive materials**: Definitions, mnemonics, examples, related words
- **RAG integration**: Automatic note creation and cross-referencing
- **Visual learning**: Memory tricks and associations
- **Progress tracking**: Focus on weak areas

### **Intelligent Notes**
- **Easy creation**: "note for [word]: [your note]"
- **Smart search**: RAG-powered similarity search
- **AI summaries**: "summarize notes about difficult words"
- **Multiple types**: User notes, mnemonics, system-generated

### **Semantic Search**
- **Concept-based**: "Find words like happiness"
- **Similarity scoring**: Ranked results with confidence
- **Cross-referencing**: Discover related vocabulary

### **Progress Analytics**
- **Real-time dashboard**: Session metrics and overall progress
- **Visual indicators**: Charts, progress bars, status breakdown
- **Recommendations**: AI-suggested study strategies
- **Session management**: Clear start/end boundaries

## 🎯 **Example Commands**

### **Quiz Commands**
```
"Start a quiz"
"Quiz me on 5 words"
"Quiz my weak words"
"Hint" (during quiz)
```

### **Study Commands**
```
"Study words"
"Help me study weak words"
"Show me mnemonics"
```

### **Notes Commands**
```
"Show notes"
"Note for sparse: means thin and scattered"
"Summarize my notes"
"Summarize notes about difficult words"
```

### **Search Commands**
```
"Find words like happiness"
"Search similar to sparse"
"Words related to anger"
```

### **Progress Commands**
```
"Show progress"
"How am I doing?"
"End session"
```

## 🔧 **Technical Features**

### **LLM Integration**
- **Model**: GPT-4o-mini for cost-effective intelligence
- **Temperature**: Optimized for different tasks (0.1 for classification, 0.7 for generation)
- **Prompt engineering**: Carefully crafted prompts for consistent results
- **Error handling**: Graceful fallbacks for API issues

### **Vector Database**
- **ChromaDB**: Persistent vector storage
- **OpenAI embeddings**: High-quality semantic representations
- **Collections**: Organized storage for notes and definitions
- **Similarity search**: Efficient nearest neighbor queries

### **Session Management**
- **State tracking**: Comprehensive session state management
- **Batch operations**: Efficient database updates
- **Context preservation**: Conversation history for better understanding
- **Progress persistence**: Reliable data storage

### **UI/UX Enhancements**
- **Streamlit interface**: Clean, responsive design
- **Interactive sidebar**: Quick actions and real-time metrics
- **Visual feedback**: Progress bars, charts, status indicators
- **Responsive design**: Works on different screen sizes

## 📊 **Performance Improvements**

### **Database Efficiency**
- **Batch updates**: Reduced database calls
- **Optimized queries**: Efficient word selection algorithms
- **Session-based operations**: Better transaction management

### **LLM Optimization**
- **Batch processing**: Multiple operations in single API calls
- **Caching**: Reduced redundant API requests
- **Smart prompting**: Efficient token usage

### **User Experience**
- **Faster responses**: Optimized processing pipeline
- **Better feedback**: More informative and encouraging
- **Intuitive interface**: Clear navigation and actions

## 🛠️ **Development**

### **Project Structure**
```
workspace/
├── app.py                    # Streamlit application
├── gre_vocab_tutor.py       # Enhanced multi-agent system
├── intent_classifier.py     # LLM-based intent classification
├── rag_system.py           # RAG implementation with ChromaDB
├── database.py             # Enhanced database operations
├── words.csv              # Vocabulary dataset (573 words)
├── requirements.txt       # Dependencies
├── test_enhanced_features.py # Test suite
└── README.md             # This file
```

### **Key Dependencies**
- **streamlit**: Web interface
- **langchain**: LLM framework
- **langgraph**: Multi-agent orchestration
- **chromadb**: Vector database
- **openai**: LLM API access

## 🎉 **Results**

The enhanced GRE Vocab Tutor now provides:

1. **Intelligent Understanding**: LLM-powered intent classification for natural interaction
2. **Rich Feedback**: Detailed quiz grading with explanations and examples
3. **Smart Learning**: RAG-powered note search and semantic word discovery
4. **Session Management**: Efficient batch updates and progress tracking
5. **Enhanced UX**: Interactive dashboard with real-time metrics
6. **Scalable Architecture**: Modular design for future enhancements

### **Before vs After**
- **Routing**: Keyword matching → LLM intent classification
- **Grading**: Simple keyword matching → Detailed LLM feedback
- **Notes**: Basic storage → RAG-powered search and summaries
- **Updates**: Immediate database writes → Session-based batch operations
- **UI**: Basic sidebar → Interactive dashboard with analytics
- **Learning**: Linear progression → Adaptive, intelligent tutoring

The application now feels significantly more intelligent, engaging, and effective for GRE vocabulary learning!