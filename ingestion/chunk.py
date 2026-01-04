"""
Split text into overlapping chunks for embedding.

Chunking is needed because long documents cannot be processed as a single
unit due to token limits and efficiency constraints. Overlap preserves context across 
chunks so meaning is not lost mid-sentence.
"""
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.
    Args:
        text (str): The input text to chunk.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list[str]: List of text chunks.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        # Move the start index forward for next chunk while preserving overlap
        start += chunk_size - overlap
    
    return chunks