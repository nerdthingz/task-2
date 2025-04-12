# task-2
Healthcare AI Content Relevance Assessment Workflow

Input Reception: The system receives a URL to a healthcare article for assessment.
Content Extraction: Using requests and BeautifulSoup, the system extracts the text content from the article URL.
Primary Analysis: The first LLM (Groq's llama3-70b) analyzes the article and:

Summarizes the article into 5 key points
Provides an initial relevance score (1-100) based on how well the content aligns with the client's "AI as enabler" philosophy


Embedding Comparison: In parallel, the system:

Creates vector embeddings of both the client's interests and the article summary using SentenceTransformer
Calculates a cosine similarity score (scaled to 0-100) between these embeddings


Comprehensive Assessment: The second LLM call receives:

The article summary
The first LLM's relevance score
The embedding similarity score
The client's interests (as system context)


Final Output Generation: The second LLM then:

Categorizes the article into one of five predefined themes
Provides a concise rationale explaining why the article would interest the physician client
Takes into account both the semantic understanding (first score) and vector similarity (embedding score)


Result Delivery: The system returns a structured response containing:

The embedding similarity score
The categorization and rationale analysis
The article summary
