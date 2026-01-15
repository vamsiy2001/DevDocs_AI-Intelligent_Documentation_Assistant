"""
Enhanced Document Loader with Multiple Framework Documentation
Includes: LangChain, LangGraph, FastAPI, Streamlit, Gradio, RAG, Vector DBs, etc.
"""

from typing import List
from langchain_core.documents import Document


class EnhancedDocumentLoader:
    """Load comprehensive documentation from multiple sources"""
    
    def __init__(self):
        print("📚 Enhanced Document Loader initialized")
    
    def load_comprehensive_docs(self) -> List[Document]:
        """
        Load comprehensive documentation covering multiple frameworks
        Perfect for a production-grade RAG system demo
        """
        print("📚 Loading comprehensive multi-framework documentation...")
        
        docs = []
        
        # ===================================================================
        # LANGCHAIN DOCUMENTATION
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# LangChain Framework Overview

LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason-driven.

Key Components:
- **Prompts**: Templates for structuring LLM inputs
- **Models**: Interfaces to various LLMs (OpenAI, Anthropic, etc.)
- **Chains**: Sequences of operations
- **Agents**: Dynamic decision-making with tools
- **Memory**: Conversation and state persistence
- **Callbacks**: Monitoring and logging

LangChain Expression Language (LCEL) allows chaining components with the pipe operator:
```python
chain = prompt | model | output_parser
result = chain.invoke({"input": "Hello"})
```

Main use cases:
- Chatbots and conversational AI
- Question answering over documents (RAG)
- Data analysis and extraction
- Code generation and analysis""",
                metadata={"source": "langchain", "topic": "overview", "framework": "langchain"}
            ),
            
            Document(
                page_content="""# LangChain Chains

Chains combine multiple components into a single application flow.

Types of Chains:
1. **LLMChain**: Basic chain with prompt + LLM
2. **Sequential Chains**: Run chains in sequence
3. **Router Chains**: Route inputs to different chains
4. **Transform Chains**: Process data between steps

Example - Simple Chain:
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms")
model = ChatOpenAI(temperature=0.7)
chain = prompt | model

result = chain.invoke({"topic": "quantum computing"})
```

Example - Sequential Chain:
```python
from langchain.chains import SequentialChain

# First chain: generate outline
outline_chain = outline_prompt | model

# Second chain: expand outline
expansion_chain = expansion_prompt | model

# Combine
full_chain = SequentialChain(
    chains=[outline_chain, expansion_chain],
    input_variables=["topic"],
    output_variables=["outline", "expanded"]
)
```

Chains can be nested and combined for complex workflows.""",
                metadata={"source": "langchain", "topic": "chains", "framework": "langchain"}
            ),
            
            Document(
                page_content="""# LangChain Agents

Agents use LLMs as reasoning engines to decide which actions to take and in what order. Unlike chains (predetermined sequence), agents dynamically choose their path.

Core Components:
- **Agent**: Decision-making LLM
- **Tools**: Functions the agent can call
- **Agent Executor**: Runs the agent loop
- **Memory**: Optional conversation history

Agent Types:
1. **Zero-shot ReAct**: Decides based on tool descriptions
2. **Conversational ReAct**: Maintains conversation context
3. **OpenAI Functions**: Uses function calling
4. **Structured Chat**: For complex tools

Example:
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Define tools
def search_tool(query: str) -> str:
    return f"Search results for: {query}"

tools = [
    Tool(name="Search", func=search_tool, description="Search the web")
]

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run
result = agent_executor.invoke({"input": "What's the weather in London?"})
```

Agents are powerful for complex, multi-step tasks requiring dynamic planning.""",
                metadata={"source": "langchain", "topic": "agents", "framework": "langchain"}
            ),
            
            Document(
                page_content="""# Retrieval Augmented Generation (RAG)

RAG enhances LLM responses by retrieving relevant information from external knowledge bases.

RAG Pipeline Steps:
1. **Load**: Import documents (PDFs, web pages, DBs)
2. **Split**: Chunk documents into smaller pieces
3. **Embed**: Convert chunks to vector embeddings
4. **Store**: Save embeddings in vector database
5. **Retrieve**: Find relevant chunks for queries
6. **Generate**: Pass chunks to LLM as context

Basic RAG Implementation:
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("docs.txt")
documents = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5. Query
result = qa_chain({"query": "What is RAG?"})
```

Advanced RAG techniques:
- Hybrid search (dense + sparse)
- Reranking retrieved results
- Query expansion
- Parent-child chunking
- Metadata filtering""",
                metadata={"source": "langchain", "topic": "rag", "framework": "langchain"}
            ),
        ])
        
        # ===================================================================
        # LANGGRAPH DOCUMENTATION
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# LangGraph - Stateful Multi-Agent Workflows

LangGraph extends LangChain with graph-based orchestration for building complex, stateful applications.

Key Concepts:
- **State**: Shared data structure passed between nodes
- **Nodes**: Functions that process state
- **Edges**: Connections between nodes (conditional or direct)
- **Graph**: Overall workflow structure

Why LangGraph?
- Build cyclic workflows (loops, retries)
- Implement agent collaboration
- Create human-in-the-loop systems
- Handle complex branching logic

Basic Example:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state
class AgentState(TypedDict):
    input: str
    output: str
    steps: int

# Define nodes
def process_node(state: AgentState):
    state["output"] = f"Processed: {state['input']}"
    state["steps"] += 1
    return state

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("process", process_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

# Compile and run
app = workflow.compile()
result = app.invoke({"input": "hello", "output": "", "steps": 0})
```

LangGraph is ideal for:
- Multi-agent systems
- Complex decision trees
- Iterative refinement workflows
- Agentic RAG systems""",
                metadata={"source": "langgraph", "topic": "overview", "framework": "langgraph"}
            ),
            
            Document(
                page_content="""# LangGraph Agent Routing

Routing in LangGraph allows dynamic decision-making between different paths.

Conditional Edges Example:
```python
def should_continue(state: AgentState) -> str:
    if state["needs_more_info"]:
        return "search"
    elif state["ready_to_answer"]:
        return "generate"
    else:
        return "process"

workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "search": "web_search_node",
        "generate": "generate_node",
        "process": "process_node"
    }
)
```

Multi-Agent Collaboration:
```python
class MultiAgentState(TypedDict):
    task: str
    researcher_output: str
    writer_output: str
    final_output: str

# Researcher agent
def researcher(state):
    state["researcher_output"] = research(state["task"])
    return state

# Writer agent
def writer(state):
    state["writer_output"] = write(state["researcher_output"])
    return state

# Build collaborative workflow
workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)
```

Routing enables intelligent, context-aware application behavior.""",
                metadata={"source": "langgraph", "topic": "routing", "framework": "langgraph"}
            ),
        ])
        
        # ===================================================================
        # VECTOR DATABASES DOCUMENTATION
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# Vector Databases Overview

Vector databases store and query high-dimensional embeddings for semantic search.

Popular Vector Databases:

**1. ChromaDB** (Local, Open Source)
- Pros: Easy setup, no server needed, great for development
- Cons: Not suitable for large-scale production
- Use case: Prototyping, small applications

**2. Pinecone** (Cloud, Managed)
- Pros: Fully managed, scalable, fast
- Cons: Paid service, vendor lock-in
- Use case: Production applications

**3. Weaviate** (Open Source, Self-hosted/Cloud)
- Pros: Feature-rich, hybrid search, GraphQL
- Cons: More complex setup
- Use case: Enterprise applications

**4. Qdrant** (Open Source, Rust-based)
- Pros: Very fast, efficient, cloud option available
- Cons: Smaller ecosystem
- Use case: Performance-critical applications

**5. FAISS** (Library, Facebook)
- Pros: Extremely fast, CPU/GPU support
- Cons: No built-in persistence, low-level
- Use case: Research, high-performance needs

Key Features to Consider:
- Scalability (millions of vectors)
- Filtering capabilities (metadata)
- Hybrid search (vector + keywords)
- Persistence and backups
- Cloud vs self-hosted
- Cost""",
                metadata={"source": "general", "topic": "vector-databases", "framework": "databases"}
            ),
            
            Document(
                page_content="""# Embeddings and Semantic Search

Embeddings convert text into numerical vectors that capture semantic meaning.

How Embeddings Work:
1. Text is tokenized into pieces
2. Neural network processes tokens
3. Output is a vector (e.g., 384, 768, or 1536 dimensions)
4. Similar texts have similar vectors (measured by cosine similarity)

Popular Embedding Models:

**OpenAI text-embedding-ada-002**
- Dimensions: 1536
- Cost: ~$0.0001 per 1K tokens
- Quality: Excellent
- Use case: Production applications

**sentence-transformers (Open Source)**
- Models: all-MiniLM-L6-v2 (384d), BGE-small (384d)
- Cost: Free (run locally)
- Quality: Good to excellent
- Use case: Cost-sensitive applications

**Cohere Embed**
- Dimensions: 1024
- Cost: Free tier available
- Quality: Excellent
- Use case: Multilingual applications

Semantic Search Process:
```python
# 1. Embed documents
doc_embeddings = embedding_model.embed_documents(documents)

# 2. Store in vector DB
vectorstore.add_embeddings(doc_embeddings)

# 3. Embed query
query_embedding = embedding_model.embed_query("user question")

# 4. Find similar vectors
results = vectorstore.similarity_search(query_embedding, k=5)
```

Similarity Metrics:
- Cosine similarity (most common)
- Euclidean distance
- Dot product""",
                metadata={"source": "general", "topic": "embeddings", "framework": "ml"}
            ),
        ])
        
        # ===================================================================
        # RAG EVALUATION DOCUMENTATION
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# RAG Evaluation with RAGAS

RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG systems.

Core Metrics:

**1. Faithfulness** (Answer grounded in context?)
- Measures if generated answer is factually consistent with retrieved context
- High score = No hallucination
- Formula: Claims in answer that are supported by context / Total claims

**2. Answer Relevancy** (Answer addresses question?)
- Measures if answer is relevant to the question
- High score = On-topic response
- Computed using similarity between question and answer

**3. Context Precision** (Retrieved docs relevant?)
- Measures if retrieved chunks are useful
- High score = No noise in retrieval
- Formula: Relevant chunks in top-k / Total chunks retrieved

**4. Context Recall** (All needed info retrieved?)
- Measures if all necessary information was retrieved
- High score = Complete retrieval
- Requires ground truth answer

Evaluation Process:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Prepare dataset
dataset = {
    "question": ["What is RAG?"],
    "answer": ["RAG is Retrieval Augmented Generation"],
    "contexts": [["RAG combines retrieval with generation"]],
    "ground_truth": ["RAG stands for Retrieval Augmented Generation"]
}

# Evaluate
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings
)

print(results)
# Output: {'faithfulness': 0.92, 'answer_relevancy': 0.88}
```

Best Practices:
- Create diverse test sets (20-50 questions)
- Test edge cases and difficult queries
- Compare configurations (A/B testing)
- Track metrics over time
- Set target thresholds (e.g., faithfulness > 0.8)""",
                metadata={"source": "ragas", "topic": "evaluation", "framework": "evaluation"}
            ),
            
            Document(
                page_content="""# Advanced RAG Techniques

Improve RAG performance with these techniques:

**1. Hybrid Search** (Dense + Sparse)
- Combine semantic search (embeddings) with keyword search (BM25)
- Best of both worlds: meaning + exact matches
- Implementation: Retrieve from both, merge results

**2. Reranking**
- Use CrossEncoder to reorder retrieved results
- More accurate than initial retrieval
- Process: Retrieve 20 chunks → Rerank → Keep top 5

**3. Query Expansion**
- Generate multiple versions of the query
- Techniques: Synonyms, HyDE (hypothetical documents)
- Increases recall

**4. Parent-Child Chunking**
- Store large chunks (parents) and small chunks (children)
- Retrieve small chunks (precise)
- Return large chunks (context)

**5. Metadata Filtering**
- Pre-filter results by metadata (date, source, category)
- Reduces search space
- Improves precision

**6. Query Routing**
- Route different questions to different retrievers
- E.g., code questions → code search, general → document search

**7. Self-Correction**
- Agent evaluates if retrieval was good
- Re-retrieves if needed
- Improves reliability

Performance Improvements:
- Hybrid search: +20-30% accuracy
- Reranking: +15-25% accuracy
- Query expansion: +10-15% recall
- Combined: +40-60% overall improvement

Implementation Example:
```python
# Hybrid + Reranking
dense_results = vectorstore.search(query, k=20)
sparse_results = bm25.search(query, k=20)
combined = merge(dense_results, sparse_results)
reranked = reranker.rerank(query, combined, top_k=5)
```""",
                metadata={"source": "general", "topic": "rag-advanced", "framework": "rag"}
            ),
        ])
        
        # ===================================================================
        # FASTAPI DOCUMENTATION
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# FastAPI Framework

FastAPI is a modern, fast web framework for building APIs with Python.

Key Features:
- Fast performance (on par with NodeJS and Go)
- Automatic API documentation (Swagger UI)
- Type hints and validation (Pydantic)
- Async support
- Easy to learn

Basic Example:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
async def ask_question(query: Query):
    answer = rag_system.query(query.question)
    return {"answer": answer}

# Run: uvicorn main:app --reload
# Docs: http://localhost:8000/docs
```

For RAG Applications:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str
    max_sources: int = 5

class Source(BaseModel):
    content: str
    score: float
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency: float

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    start = time.time()
    
    result = rag.query(request.question)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency=time.time() - start
    )
```

Benefits for RAG:
- Easy API deployment
- Automatic validation
- Built-in documentation
- Async for better performance""",
                metadata={"source": "fastapi", "topic": "overview", "framework": "fastapi"}
            ),
        ])
        
        # ===================================================================
        # GRADIO DOCUMENTATION
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# Gradio for ML Interfaces

Gradio creates web UIs for machine learning models with minimal code.

Why Gradio?
- Deploy in 3 lines of code
- Automatic UI generation
- Share with public links
- Free hosting on HuggingFace Spaces

Basic Example:
```python
import gradio as gr

def predict(text):
    return f"You said: {text}"

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
```

RAG Application:
```python
import gradio as gr

def rag_query(question, history):
    result = rag.query(question)
    return result["answer"]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    clear = gr.Button("Clear")
    
    msg.submit(rag_query, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot)

demo.launch(share=True)
```

Features:
- Chatbot interface
- File uploads
- Real-time updates
- Custom themes
- Authentication
- Analytics

Deployment:
- HuggingFace Spaces (free)
- Share links (public/private)
- Embed in websites

Perfect for:
- Demos and prototypes
- Internal tools
- Quick deployments""",
                metadata={"source": "gradio", "topic": "overview", "framework": "gradio"}
            ),
        ])
        
        # ===================================================================
        # PROMPT ENGINEERING
        # ===================================================================
        
        docs.extend([
            Document(
                page_content="""# Prompt Engineering Best Practices

Effective prompts are crucial for good LLM outputs.

Key Techniques:

**1. Be Specific and Clear**
Bad: "Explain this"
Good: "Explain quantum entanglement in 2-3 sentences for a high school student"

**2. Provide Context**
```
Context: You are a Python expert helping beginners.
Task: Explain list comprehensions.
Format: Use simple language and provide 2 examples.
```

**3. Use Examples (Few-Shot)**
```
Q: What is the capital of France?
A: The capital of France is Paris.

Q: What is the capital of Japan?
A: The capital of Japan is Tokyo.

Q: What is the capital of Brazil?
A:
```

**4. Chain of Thought**
```
Think step by step:
1. First, identify the main components
2. Then, explain how they interact
3. Finally, summarize the key points
```

**5. Constrain Output Format**
```
Respond in JSON format:
{
  "summary": "...",
  "key_points": ["...", "..."],
  "confidence": 0.0-1.0
}
```

**6. Role Assignment**
```
You are an expert data scientist. Analyze this dataset and provide insights.
```

For RAG Systems:
```
Use the following context to answer the question.
If the context doesn't contain the answer, say "I don't have enough information."
Always cite which source you used.

Context:
{context}

Question: {question}

Answer:
```

Prompt Templates in LangChain:
```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
```""",
                metadata={"source": "general", "topic": "prompt-engineering", "framework": "prompting"}
            ),
        ])
        
        print(f"✅ Loaded {len(docs)} comprehensive documents")
        print(f"   Frameworks covered: LangChain, LangGraph, FastAPI, Gradio, RAGAS")
        print(f"   Topics: RAG, Embeddings, Vector DBs, Agents, Evaluation, Prompting")
        
        return docs


# Example usage
if __name__ == "__main__":
    loader = EnhancedDocumentLoader()
    docs = loader.load_comprehensive_docs()
    
    print(f"\nSample document:")
    print(f"Topic: {docs[0].metadata['topic']}")
    print(f"Framework: {docs[0].metadata['framework']}")
    print(f"Content preview: {docs[0].page_content[:200]}...")