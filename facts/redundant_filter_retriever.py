from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    # Objects for the class - no need to "hard-code" them
    # in the below functions
    embeddings: Embeddings
    chroma: Chroma

    def _get_relevant_documents(self, query: str): 
        '''
        Docstring for get_relevant_documents:
        1.Calculate embeddings for 'query' string
        2.takes embeddings and feed them into max_marginal_relevance_search_by_vector
        '''
        emb = self.embeddings.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            k=4,
            fetch_k=60,
            embedding=emb,
            lambda_mult=0.8,
        )
    
    async def aget_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)



    