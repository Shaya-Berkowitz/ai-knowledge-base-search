import { ChangeEvent, useState } from 'react'
import './App.css'

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function askRag(queryText: string){
    const response = await fetch('http://127.0.1:8000/rag-answer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: queryText, top_k: 5, score_threshold: 0.3 }),
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.statusText}`);
    }

    return response.json();
  }

  async function handleAsk(){
    console.log('handleAsk clicked');
    if (!query.trim()) {
        setError("Please enter a question.");
        return;
      }

    setError("");
    setLoading(true);

    try {
      const data = await askRag(query);
      setAnswer(data.answer ?? "No answer returned.");
    } catch (err) {
      setError("Failed to get answer.");
      setAnswer("");
    } finally {
      setLoading(false);
    }

  }


  return (
    <main>
        <h1>AI KNOWLEDGE PLATFORM</h1>
        <h3>Upload your documents and ask OpenAI whatever you want about them.</h3>

        <label>Your Question</label>
        <input id="question-input" type="text" value={query} onChange={(e: ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)} placeholder='Ask something about your docs...'/>

        
        <button onClick={handleAsk} disabled={loading || !query.trim()}>
          {loading ? "Asking..." : "Ask Question"}
        </button>

        {error && <p>{error}</p>}

        <h2>Your Answer</h2>
        <p>{answer || "No answer yet."}</p>

        <h2>Sources</h2>
        <p>No citations yet.</p>

        

    </main>
  )
}

export default App
