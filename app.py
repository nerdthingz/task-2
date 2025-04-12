import requests
from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
model = SentenceTransformer('all-MiniLM-L6-v2')

CLIENT_INTERESTS = """
AI should enable healthcare professionals, not replace them.
Focus on workflow optimization and physician experience.
Interest in AI tools that reduce administrative burden.
Emphasis on patient-centered care supported by technology.
Value AI solutions that augment clinical decision-making.
"""

def extract_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return ' '.join([p.text for p in soup.find_all('p')])

def assess_relevance(article_url):
    article_text = extract_article_content(article_url)
    
    summary_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": f"Your aim is to extract the core concepts to assess relevance to healthcare AI. Summarize this healthcare article in 5 key points, then rate on a scale of 1-100 how relevant this article is to a physician who believes 'AI should enable healthcare professionals, not replace them':\n\n{article_text[:5000]}"}]
    )
    article_summary = summary_response.choices[0].message.content
    
    client_embedding = model.encode(CLIENT_INTERESTS)
    summary_embedding = model.encode(article_summary)
    embedding_score = float(np.dot(client_embedding, summary_embedding) / 
                         (np.linalg.norm(client_embedding) * np.linalg.norm(summary_embedding))) * 100
    
    categorization = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": CLIENT_INTERESTS},
            {"role": "user", "content": f"Your aim is to categorize content and explain relevance. Article has two scores: AI-generated relevance score found in the summary AND a similarity score of {embedding_score:.1f}/100 based on embedding comparison. Categorize this article into one of: 'AI Ethics', 'Clinical Applications', 'Workflow Optimization', 'Patient Experience', 'Healthcare Policy'. Then provide a 2-sentence rationale for why this might interest a physician with the above perspective.\n\nArticle summary and first relevance score: {article_summary}"}
        ]
    )
    
    return {"embedding_score": embedding_score, "analysis": categorization.choices[0].message.content, "summary": article_summary}
