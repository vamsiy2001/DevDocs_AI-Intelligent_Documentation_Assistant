"""
LangGraph RAG Agent
- Routes between RAG, Web Search, and Direct Answer
- Implements self-correction and multi-step reasoning
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, LLM_CONFIG


class AgentState(TypedDict):
    """State of the agent"""
    question: str
    documents: List[Dict[str, Any]]
    generation: str
    route: str
    iterations: int
    search_query: str
    needs_refinement: bool


class SimpleRAG:
    """
    RAG without complex routing
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
        
        print("🤖 Initializing SimpleRAG...")
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            **LLM_CONFIG
        )
        print("✅ SimpleRAG ready!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Simple RAG query"""
        
        # 1. Retrieve
        print(f"📚 Retrieving for: {question}")
        documents = self.retriever.hybrid_search(question)
        
        if not documents:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "num_sources": 0
            }
        
        # 2. Format context
        context = "\n\n".join([
            f"[Source {i+1}] (Relevance: {doc['score']:.2f})\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # 3. Generate
        prompt = f"""Use the following context to answer the question accurately and concisely.
If the context doesn't contain enough information, say so honestly.
Always cite which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": documents,
                "num_sources": len(documents)
            }
        
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": documents,
                "num_sources": len(documents)
            }


class RAGAgent:
    """
    Advanced RAG Agent with LangGraph
    Routes queries intelligently and performs multi-step reasoning
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
        
        print("🤖 Initializing RAG Agent with LangGraph...")
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            **LLM_CONFIG
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("✅ RAG Agent ready!")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.web_search)
        
        # Set entry point
        workflow.set_entry_point("route_query")
        
        # Add edges
        workflow.add_conditional_edges(
            "route_query",
            self.decide_route,
            {
                "retrieve": "retrieve",
                "web_search": "web_search",
                "direct": "generate"
            }
        )
        
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self.check_relevance,
            {
                "generate": "generate",
                "web_search": "web_search",
                "refine": "route_query"
            }
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def route_query(self, state: AgentState) -> AgentState:
        """
        Route the query to appropriate path
        """
        question = state["question"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing expert. Classify the query into one of these categories:

- 'retrieve': Technical questions about specific tools, libraries, or documentation
- 'web_search': Questions about recent events, news, or current information
- 'direct': Simple questions that can be answered from general knowledge

Examples:
Q: "How to use LangChain agents?" → retrieve
Q: "What happened in tech news today?" → web_search  
Q: "What is Python?" → direct

Respond with ONLY the category name."""),
            ("human", f"Question: {question}\n\nCategory:")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            route = response.content.strip().lower()
            
            print(f"🔀 Route: {route}")
            
            state["route"] = route
            state["iterations"] = 0
            
        except Exception as e:
            print(f"⚠️ Error in routing: {e}, defaulting to retrieve")
            state["route"] = "retrieve"
            state["iterations"] = 0
        
        return state
    
    def decide_route(self, state: AgentState) -> str:
        """Decide which path to take"""
        route = state["route"]
        
        if "retrieve" in route or "rag" in route:
            return "retrieve"
        elif "web" in route or "search" in route:
            return "web_search"
        else:
            return "direct"
    
    def retrieve(self, state: AgentState) -> AgentState:
        """Retrieve documents using hybrid search"""
        question = state["question"]
        
        print(f"📚 Retrieving documents for: {question}")
        
        try:
            documents = self.retriever.hybrid_search(question)
            state["documents"] = documents
            print(f"✅ Retrieved {len(documents)} documents")
        
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            state["documents"] = []
        
        return state
    
    def grade_documents(self, state: AgentState) -> AgentState:
        """
        Grade retrieved documents for relevance
        Simple version: just check scores
        """
        documents = state["documents"]
        
        print("🎯 Grading document relevance...")
        
        if not documents:
            return state
        
        # Simple grading: filter by score threshold
        threshold = 0.3
        relevant_docs = [doc for doc in documents if doc["score"] > threshold]
        
        state["documents"] = relevant_docs
        print(f"✅ {len(relevant_docs)}/{len(documents)} documents passed threshold")
        
        return state
    
    def check_relevance(self, state: AgentState) -> str:
        """Check if we have relevant documents"""
        documents = state["documents"]
        iterations = state["iterations"]
        
        if not documents:
            print("⚠️ No relevant documents found")
            if iterations < 1:
                return "web_search"
            else:
                return "generate"
        
        # Check quality of top document
        if documents:
            top_score = documents[0]["score"]
            
            if top_score < 0.5 and iterations < 1:
                print(f"⚠️ Low relevance score: {top_score:.2f}, trying web search")
                state["iterations"] += 1
                return "web_search"
        
        return "generate"
    
    def web_search(self, state: AgentState) -> AgentState:
        """
        Fallback to web search
        Placeholder - integrate Tavily or other search API here
        """
        question = state["question"]
        
        print(f"🌐 Web search fallback for: {question}")
        
        # Placeholder: Add actual web search integration
        state["documents"] = [
            {
                "content": f"Web search results would appear here for: {question}",
                "metadata": {"source": "web_search"},
                "score": 0.7
            }
        ]
        
        return state
    
    def generate(self, state: AgentState) -> AgentState:
        """Generate answer using retrieved documents"""
        question = state["question"]
        documents = state["documents"]
        
        print("✍️ Generating answer...")
        
        if not documents:
            state["generation"] = "I don't have enough information to answer this question accurately."
            return state
        
        # Format context from documents
        context = "\n\n".join([
            f"[Source {i+1}] (Relevance: {doc['score']:.2f})\n{doc['content']}"
            for i, doc in enumerate(documents[:3])
        ])
        
        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful technical documentation assistant.
Use the provided context to answer the question accurately and concisely.
If the context doesn't contain enough information, say so honestly.
Always cite which sources you used.

Context:
{context}"""),
            ("human", "{question}")
        ])
        
        try:
            response = self.llm.invoke(
                prompt.format_messages(context=context, question=question)
            )
            
            state["generation"] = response.content
        
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            state["generation"] = f"Error generating answer: {str(e)}"
        
        return state
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the agent on a question
        Returns: answer, source documents, and metadata
        """
        initial_state = {
            "question": question,
            "documents": [],
            "generation": "",
            "route": "",
            "iterations": 0,
            "search_query": "",
            "needs_refinement": False
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result["generation"],
                "documents": result["documents"],
                "route": result["route"],
                "iterations": result["iterations"]
            }
        
        except Exception as e:
            print(f"❌ Error running agent: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "documents": [],
                "route": "error",
                "iterations": 0
            }


# Example usage
if __name__ == "__main__":
    print("Testing agents requires a retriever instance")