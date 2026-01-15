import gradio as gr
from typing import List, Dict, Tuple
import time
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import HybridRetriever
from src.agents import SimpleRAG
from src.config import settings


class DevDocsApp:
    """Main Gradio application"""
    
    def __init__(self):
        print("🚀 Initializing DevDocs AI App...")
        
        try:
            # Initialize RAG system
            self.retriever = HybridRetriever()
            
            # Load existing vector store first
            loaded = self.retriever.load_vector_store()
            
            if loaded:
                # Vector store loaded successfully
                self.rag = SimpleRAG(self.retriever)
                self.initialized = True
                print("✅ DevDocs AI ready with existing vector store!")
            else:
                # If no vector store, load sample docs
                print("⚠️  No vector store found, loading sample data...")
                from src.ingestion.document_loader import DocumentLoader
                
                loader = DocumentLoader()
                docs = loader.load_sample_docs()
                
                from src.ingestion.chunking import DocumentChunker
                chunker = DocumentChunker()
                chunks = chunker.recursive_character_split(docs)
                
                self.retriever.create_vector_store(chunks)
                self.rag = SimpleRAG(self.retriever)
                self.initialized = True
                print("✅ DevDocs AI ready with sample data!")
        
        except Exception as e:
            print(f"❌ Error initializing app: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False
            self.error_message = str(e)
        
        # Statistics
        self.query_count = 0
        self.start_time = datetime.now()
    
    def query_rag(
        self, 
        question: str, 
        history: List[Dict]
    ) -> tuple:
        """
        Process a query and return answer with sources
        
        Args:
            question: User's question
            history: Chat history in Gradio 6.0 format [{"role": "user", "content": "..."}, ...]
        
        Returns:
            Tuple of (updated_history, sources_html)
        """
        if not self.initialized:
            error_msg = "❌ System not initialized. Please check the logs."
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        if not question.strip():
            return history, ""
        
        self.query_count += 1
        
        # Track time
        start = time.time()
        
        try:
            # Query RAG
            result = self.rag.query(question)
            
            latency = time.time() - start
            
            # Format answer
            answer = result["answer"]
            
            # Format sources
            sources_html = self._format_sources(
                result["sources"], 
                latency,
                self.query_count
            )
            
            # Add to history in Gradio 6.0 format
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            
            return history, sources_html
        
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def _format_sources(
        self, 
        sources: List[dict], 
        latency: float,
        query_num: int
    ) -> str:
        """Format sources as HTML"""
        
        html = f"""
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 15px; border: 1px solid #dee2e6;'>
            <h3 style='margin-top: 0; color: #495057;'>📚 Sources & Metadata</h3>
            
            <div style='margin-bottom: 20px; padding: 10px; background: white; border-radius: 5px;'>
                <span style='margin-right: 20px;'><strong>⚡ Latency:</strong> {latency:.2f}s</span>
                <span style='margin-right: 20px;'><strong>📊 Query #:</strong> {query_num}</span>
                <span><strong>🎯 Sources Used:</strong> {len(sources)}</span>
            </div>
        """
        
        if not sources:
            html += "<p><em>No sources retrieved</em></p>"
        else:
            for i, source in enumerate(sources[:3], 1):
                score = source.get('score', 0)
                content = source.get('content', '')[:300]
                metadata = source.get('metadata', {})
                
                # Color based on relevance score
                if score > 0.7:
                    border_color = "#28a745"  # Green
                elif score > 0.5:
                    border_color = "#ffc107"  # Yellow
                else:
                    border_color = "#6c757d"  # Gray
                
                html += f"""
                <div style='background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 5px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-weight: bold; color: {border_color}; margin-bottom: 8px;'>
                        📄 Source {i} (Relevance: {score:.2%})
                    </div>
                    <div style='margin: 8px 0; font-size: 0.95em; color: #212529; line-height: 1.6;'>
                        {content}...
                    </div>
                    <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #e9ecef; font-size: 0.85em; color: #6c757d;'>
                        <span style='margin-right: 15px;'>📁 {metadata.get('source', 'Unknown')}</span>
                        <span>🏷️ {metadata.get('topic', 'N/A')}</span>
                    </div>
                </div>
                """
        
        html += "</div>"
        
        return html
    
    def get_stats(self) -> str:
        """Get system statistics"""
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        stats = f"""
### 📊 System Statistics

- **Uptime:** {int(hours)}h {int(minutes)}m {int(seconds)}s
- **Total Queries:** {self.query_count}
- **LLM Model:** {settings.LLM_MODEL}
- **Embedding Model:** {settings.EMBEDDING_MODEL.split('/')[-1]}
- **Vector DB:** {settings.VECTOR_DB.upper()}
- **Reranking:** {'✅ Enabled' if settings.USE_RERANKING else '❌ Disabled'}
- **Status:** {'🟢 Running' if self.initialized else '🔴 Error'}
        """
        
        return stats
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        # Create blocks without theme/css (moved to launch in Gradio 6.0)
        with gr.Blocks(title="DevDocs AI - Intelligent Documentation Assistant") as demo:
            
            gr.Markdown("""
            # 🤖 DevDocs AI - Intelligent Documentation Assistant
            
            Ask questions about developer tools, frameworks, and libraries!
            
            **Features:**
            - 🔍 Hybrid Search (Dense + Sparse + Reranking)
            - 🎯 Context-aware answers with sources
            - ⚡ Powered by Groq (Free, Fast)
            - 📊 Real-time metrics
            
            ---
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="💬 Chat History",
                        height=400,
                        show_label=True,
                        avatar_images=("👤", "🤖")
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Ask a question",
                            placeholder="e.g., How do I use LangChain agents?",
                            lines=2,
                            scale=4
                        )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Ask", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
                    
                    # Sources display
                    sources_display = gr.HTML(label="📚 Sources & Metadata")
                
                with gr.Column(scale=1):
                    # Sidebar
                    gr.Markdown("### ⚙️ System Status")
                    
                    stats_display = gr.Markdown(self.get_stats())
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                    
                    gr.Markdown("""
                    ### 💡 Example Questions
                    
                    - What is LangChain?
                    - How to create a chain?
                    - Explain RAG in simple terms
                    - What are agents?
                    - How do embeddings work?
                    """)
                    
                    gr.Markdown("""
                    ### 🛠️ Tech Stack
                    
                    - **LLM:** Groq (Llama 3.3 70B)
                    - **Embeddings:** sentence-transformers
                    - **Vector DB:** ChromaDB
                    - **Reranking:** CrossEncoder
                    - **Framework:** LangChain
                    
                    ✅ 100% Free Tools!
                    """)
            
            # Event handlers
            def respond(message, chat_history):
                """Handle user message and generate response"""
                if not message.strip():
                    return chat_history, ""
                
                # Query RAG and get updated history
                updated_history, sources_html = self.query_rag(message, chat_history)
                
                return updated_history, sources_html
            
            def clear_chat():
                """Clear chat history"""
                return [], ""
            
            submit_btn.click(
                respond,
                inputs=[question_input, chatbot],
                outputs=[chatbot, sources_display]
            ).then(
                lambda: "",  # Clear input after submission
                outputs=[question_input]
            )
            
            question_input.submit(
                respond,
                inputs=[question_input, chatbot],
                outputs=[chatbot, sources_display]
            ).then(
                lambda: "",  # Clear input after submission
                outputs=[question_input]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, sources_display]
            )
            
            refresh_stats_btn.click(
                self.get_stats,
                outputs=[stats_display]
            )
            
            gr.Markdown("""
            ---
            
            ### 📚 About This Project
            
            This is a production-grade RAG system built with 100% free tools:
            - Advanced retrieval with hybrid search and reranking
            - Evaluated using RAGAS framework
            - Deployed on Hugging Face Spaces
            
            Made with ❤️ for learning
            """)
        
        return demo


# Initialize and launch
if __name__ == "__main__":
    app = DevDocsApp()
    demo = app.create_interface()
    
    # Launch with theme and css (Gradio 6.0 format)
    demo.launch(
        share=False,  # Set to True to create public link
        server_name=settings.GRADIO_SERVER_NAME,
        server_port=settings.GRADIO_SERVER_PORT,
        show_error=True
    )