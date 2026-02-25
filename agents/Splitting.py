import fitz

class DocSplitterAgent:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size= chunk_size
        self.chunk_overlap= chunk_overlap

    def extract_text(self, pdf_path):
        doc= fitz.open(pdf_path)
        text=""
        for page in doc:
            text += page.get_text("text")
        return text
    
    def split_text(self, text):
        chunks=[]
        start=0
        while start<len(text):
            end= start+self.chunk_size
            chunk= text[start:end]
            chunks.append(chunk)
            start+= self.chunk_size - self.chunk_overlap
        return chunks
    
if __name__=="__main__":
    agent= DocSplitterAgent()
    text= agent.extract_text("/Users/lakshminishevitha/Downloads/DSE501_Project_Proposal_Group42.pdf")
    chunks= agent.split_text(text)
    print(f"Extracted {len(chunks)} chunks")    
    