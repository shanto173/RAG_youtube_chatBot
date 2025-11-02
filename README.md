# RAG_youtube_chatBot

# YouTube Video RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables interactive conversations about YouTube video content using LangChain, OpenAI, and FAISS.

## Overview

This project implements a complete RAG pipeline that:
1. Extracts transcripts from YouTube videos
2. Processes and indexes the content using vector embeddings
3. Retrieves relevant context based on user queries
4. Generates accurate responses using GPT-4o-mini

## Features

- 🎥 **YouTube Transcript Extraction**: Automatically fetches video transcripts
- 🔍 **Semantic Search**: Uses FAISS vector store for efficient similarity search
- 🤖 **Context-Aware Responses**: GPT-4o-mini generates answers based on retrieved context
- ⛓️ **LangChain Integration**: Leverages LangChain's composable chains
- 📝 **Structured Outputs**: Supports various output formats (text, lists, etc.)

## Architecture

The RAG pipeline consists of four main stages:

### 1. **Indexing** (Steps 1a-1d)
- **Document Ingestion**: Fetch YouTube transcript using `youtube-transcript-api`
- **Text Splitting**: Chunk transcript into manageable pieces (1000 chars, 200 overlap)
- **Embedding Generation**: Convert chunks to vectors using OpenAI embeddings
- **Vector Storage**: Store embeddings in FAISS for fast retrieval

### 2. **Retrieval** (Step 2)
- Configure similarity search to retrieve top 4 relevant chunks
- Uses cosine similarity for matching queries to context

### 3. **Augmentation** (Step 3)
- Combines user query with retrieved context
- Uses custom prompt template to instruct the LLM

### 4. **Generation** (Step 4)
- GPT-4o-mini generates contextual responses
- Parses output as structured text

## Installation

```bash
# Install required packages
pip install numpy pandas pydantic scikit-learn xgboost imbalanced-learn \
            langchain openai tqdm langchain_openai youtube-transcript-api \
            langchain-community faiss-cpu tiktoken python-dotenv
```

## Configuration

### Google Colab Setup

```python
from google.colab import userdata
import os

# Load OpenAI API key from Colab secrets
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
```

### Local Environment Setup

```python
from dotenv import load_dotenv
import os

load_dotenv()
# Ensure OPENAI_API_KEY is set in your .env file
```

## Usage

### Basic Example

```python
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# 1. Extract transcript
VIDEO_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
video_id = VIDEO_URL.split("v=")[1].split("&")[0]
transcript_data = YouTubeTranscriptApi.fetch(video_id, languages=["en"])
transcript = " ".join([snippet.text for snippet in transcript_data])

# 2. Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# 3. Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 4}
)

# 4. Build chain
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from the provided transcript context.
    If the content is insufficient, say "I don't know."
    
    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 5. Query
question = "What is the main topic discussed?"
docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in docs)
response = llm.invoke(prompt.format(context=context, question=question))
print(response.content)
```

### Advanced: Using LangChain Chains

```python
from langchain_core.runnables import (
    RunnableParallel, 
    RunnablePassthrough, 
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Parallel processing chain
parallel_chain = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnablePassthrough()
)

# Complete RAG chain
rag_chain = parallel_chain | prompt | llm | StrOutputParser()

# Query with structured output
answer = rag_chain.invoke("Explain the key concepts in a list format")
print(answer)
```

## Example Queries

Based on the Lex Friedman & Demis Hassabis conversation:

```python
# Who is Demis Hassabis?
rag_chain.invoke("Who is Demis Hassabis?")

# Technical questions
rag_chain.invoke("What games did Demis work on?")

# Philosophical questions
rag_chain.invoke("What are Demis's views on alien civilizations?")

# Formatted responses
rag_chain.invoke("List the key AI achievements mentioned, in bullet points")
```

## Key Components

### Vector Store Configuration
```python
# Similarity search with top-k results
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### Prompt Template
```python
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from provided transcript context.
    If content is insufficient, say "I don't know."
    
    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)
```

### LLM Configuration
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2  # Lower temperature for more focused responses
)
```

## Performance Tips

1. **Chunk Size**: Adjust based on content complexity
   - Technical content: 500-800 chars
   - Conversational: 1000-1500 chars

2. **Overlap**: Maintain context continuity
   - Standard: 200 chars
   - Dense content: 300-400 chars

3. **Top-K Retrieval**: Balance relevance vs context
   - Simple queries: k=2-3
   - Complex queries: k=4-6

4. **Temperature**: Control response creativity
   - Factual answers: 0.0-0.3
   - Creative responses: 0.7-1.0

## Limitations

- Requires valid YouTube transcript (captions must be available)
- OpenAI API costs apply for embeddings and generation
- Context window limitations (~8k tokens for gpt-4o-mini)
- Transcript quality affects answer accuracy

## Dependencies

- `langchain`: ^0.3.27
- `langchain-openai`: ^0.3.35
- `langchain-community`: ^0.3.31
- `openai`: ^1.109.1
- `faiss-cpu`: ^1.12.0
- `youtube-transcript-api`: ^1.2.3
- `tiktoken`: ^0.12.0

## Project Structure

```
youtube-rag-chatbot/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── notebooks/
│   └── langchain_youtube_chatBot.ipynb
│
├── src/
│   ├── __init__.py
│   ├── transcript_extractor.py
│   ├── vector_store.py
│   └── rag_chain.py
│
└── tests/
    └── test_rag.py
```

## Future Enhancements

- [ ] Support for multi-video conversations
- [ ] Conversation history/memory
- [ ] Web interface (Streamlit/Gradio)
- [ ] Support for additional video platforms
- [ ] Custom embedding models
- [ ] Hybrid search (keyword + semantic)
- [ ] Caching mechanism for repeated queries
- [ ] Multi-language support

## Troubleshooting

### Common Issues

**Transcript Not Available**
```python
try:
    transcript = YouTubeTranscriptApi.fetch(video_id, languages=["en", "hi"])
except TranscriptsDisabled:
    print("Captions are disabled for this video")
```

**API Rate Limits**
- Implement exponential backoff
- Cache embeddings locally
- Use batch processing for multiple videos

**Memory Issues**
- Reduce chunk size
- Process videos in segments
- Use streaming for large transcripts

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Acknowledgments

- Example uses Lex Fridman's interview with Demis Hassabis
- Built with LangChain and OpenAI
- FAISS by Meta Research

## Support

For issues or questions:
- Open a GitHub issue
- Check LangChain documentation: https://docs.langchain.com
- OpenAI API docs: https://platform.openai.com/docs

---

**Note**: Remember to keep your API keys secure and never commit them to version control. Use environment variables or secrets management.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run the notebook
jupyter notebook notebooks/langchain_youtube_chatBot.ipynb
```