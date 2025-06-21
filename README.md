# GRE Vocabulary Tutor

An interactive application to help you learn and master GRE vocabulary words using spaced repetition and active recall techniques.

## Features

- **Interactive Quizzes**: Test your knowledge with multiple-choice questions
- **Smart Spaced Repitition**: Words are scheduled based on your performance
- **Word Status Tracking**: Track words as Unknown, Weak, Moderate, or Strong
- **Study Mode**: Focus on words you need to learn the most
- **Notes**: Add personal notes to words for better retention
- **Progress Tracking**: Monitor your vocabulary growth over time

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gre-vocab-learner.git
   cd gre-vocab-learner
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. Initialize the database:
   ```bash
   python -c "from database import init_database; init_database()"
   ```

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to the URL shown in the terminal (usually http://localhost:8501).

## Usage

### Taking a Quiz

1. Click "Start New Quiz" in the sidebar
2. Read the word and select the correct definition
3. Review the correct answer and add notes if needed
4. Click "Next Question" to continue

### Studying Words

1. Click "Study Words" in the sidebar
2. Browse through words and their definitions
3. Expand each word to see examples and add your own notes
4. Track your progress with the mastery percentage in the sidebar

### Adding Notes

1. While in quiz or study mode, find the word you want to add a note to
2. Click on the word to expand it
3. Type your note in the text area
4. Click "Save Note" to store it

## How It Works

The application uses a spaced repetition algorithm to help you learn more efficiently:

- **Unknown**: Words you haven't seen or struggled with
- **Weak**: Words you've gotten wrong more than right
- **Moderate**: Words you're starting to recognize
- **Strong**: Words you consistently get right

The system prioritizes words you know less well while occasionally reviewing stronger words to reinforce your memory.

## Data

The application comes with a default set of GRE vocabulary words in `words.csv`. You can customize this file to add your own words or modify existing ones.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI](https://openai.com/)
- Inspired by effective language learning techniques

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