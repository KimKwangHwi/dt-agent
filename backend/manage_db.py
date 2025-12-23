import os
import json
import faiss
import numpy as np
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# --- 상수 정의 ---
DB_PATH = Path("data/faiss_db.json")
DB_PERMANENT_PATH = Path("data/faiss_db_permanent.json")
INDEX_PATH = Path("data/faiss_index.bin")

# 임베딩 모델 정보
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
EMBEDDING_DIMENSION = 768

def clear_db():
    """
    Deletes the FAISS database and index files to reset the vector store.
    """
    print("Attempting to clear the vector database...")
    files_to_delete = [DB_PATH, INDEX_PATH, DB_PERMANENT_PATH]
    
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"  - Successfully deleted: {file_path}")
            except OSError as e:
                print(f"  - Error deleting {file_path}: {e}")
        else:
            print(f"  - {file_path} not found. Nothing to delete.")
            
    print("\nDatabase clearing process finished.")

def rebuild_db():
    """
    Rebuilds the FAISS index from the faiss_db.json file.
    """
    print("Attempting to rebuild the FAISS index from faiss_db.json...")

    # 1. DB 파일 존재 여부 확인
    if not DB_PATH.exists():
        print(f"Error: Database file not found at '{DB_PATH}'.")
        print("Please create the faiss_db.json file first.")
        return

    # 2. DB 파일 로드
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{DB_PATH}'. Please check the file format.")
        return
        
    if not db:
        print("Warning: faiss_db.json is empty. A new, empty index will be created.")
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    else:
        questions = list(db.keys())
        print(f"Found {len(questions)} questions in faiss_db.json.")

        # 3. 임베딩 모델 초기화
        print(f"Initializing embedding model: {EMBEDDING_MODEL}")
        try:
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            print("Please ensure you have an internet connection and the required packages are installed.")
            return

        # 4. 모든 질문을 임베딩
        print("Embedding questions... (This may take a moment)")
        question_embeddings = embedding_model.embed_documents(questions)
        
        # 5. FAISS 인덱스 생성 및 데이터 추가
        print("Creating new FAISS index...")
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        index.add(np.array(question_embeddings, dtype="float32"))

    # 6. 인덱스 파일 저장
    try:
        faiss.write_index(index, str(INDEX_PATH))
        print(f"\nSuccessfully rebuilt index and saved to '{INDEX_PATH}'.")
        print(f"Total items in index: {index.ntotal}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")


if __name__ == "__main__":
    # 실행하려는 함수를 아래에 직접 호출하세요.
    # 예: DB를 초기화하려면 clear_db()를, DB를 재구성하려면 rebuild_db()를 호출하세요.
    
    # 기본적으로 DB 재구성 실행
    # rebuild_db()
    
    # DB 초기화를 원할 경우 아래 줄의 주석을 해제하고 rebuild_db()를 주석 처리하세요.
    clear_db()
