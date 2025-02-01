from .parser import Parser
from .indexer import KnowledgeGraphIndexer, VectorStoreIndexer
import logging

# Configure logging
logger = logging.getLogger(__name__)

# documents = "# Large Language Models (LLMs)\n\nLarge Language Models (LLMs) operate using advanced neural network architectures, primarily transformers, which revolutionized natural language processing (NLP). Transformers use mechanisms called self-attention and positional encoding to process input sequences as a whole, rather than word by word. This approach allows LLMs to understand the context of a word within a broader sentence or even across multiple sentences.\n\nSelf-attention computes the importance of every word relative to others in a sequence, enabling the model to capture relationships such as grammar, semantics, and long-range dependencies. For example, in the sentence 'The dog that chased the cat barked,' self-attention helps the model associate 'barked' with 'dog,' despite the intervening words.\n\n# Training Process\n\nThe training process of an LLM involves two critical steps: pretraining and fine-tuning. During pretraining, the model learns general language patterns from vast datasets, often including books, websites, and other large text corpora. This phase typically uses unsupervised learning objectives like masked language modeling (MLM), where certain words in a sentence are hidden, and the model predicts them, or causal language modeling (CLM), where the model predicts the next word in a sequence.\n\nThe model's billions of parameters are adjusted iteratively through gradient descent, optimizing its ability to minimize prediction errors. Fine-tuning narrows the model's focus by exposing it to task-specific datasets, such as sentiment analysis or legal documents, allowing it to adapt its.\n---\n# Pretrained Knowledge\n\nto specific domains.\n\nOnce trained, LLMs rely on tokenization to process textual input. Tokenization breaks text into smaller units, such as words or subwords, which are then converted into numerical representations called embeddings.\n\nThese embeddings are processed in multiple layers of the transformer architecture, where each layer refines the information by applying self-attention and feed-forward networks. The output of the final layer generates a probability distribution over possible next tokens, enabling the model to generate coherent and context-aware text.\n\nDespite their efficiency, LLMs require substantial computational resources for both training and inference, as well as mechanisms like model pruning, quantization, or distillation to make them more scalable and accessible for real-world applications."

class Indexer:
    def __init__(self):
        logger.debug("Initializing Indexer")
        self.knowledge_graph_indexer = KnowledgeGraphIndexer()
        self.vector_store_indexer = VectorStoreIndexer()
        # self.analytical_indexer = AnalyticalIndexer()

    async def index(self, file, index_name):
        logger.info(f"Starting indexing process for file with index name: {index_name}")
        try:
            logger.debug(f"Attempting to parse file using Parser.load_documents")
            file_name = file.filename    
            parsed_results = await Parser.load_data(file)
            
            if "parsed_content" not in parsed_results:
                logger.error(f"Error parsing file: {parsed_results['error']}")
                return False
            
            documents = parsed_results.get("parsed_content")
            logger.debug(f"Successfully parsed documents from file: {file_name}")

            logger.debug(f"Indexing documents in knowledge graph with index: {index_name}")
            kg_status = self.knowledge_graph_indexer.index(index_name, documents)

            logger.debug(f"Indexing documents in vector store with file: {file_name}, index: {index_name}")
            vector_status = self.vector_store_indexer.index(file_name, index_name, documents)
            
            if kg_status and vector_status:
                logger.info(f"Successfully indexed file {file_name}")
                return True
            else:
                logger.error(f"Failed to index file {file_name}")
                return False

        except Exception as e:
            logger.error(f"Error indexing file: {str(e)}", exc_info=True)
            return False
