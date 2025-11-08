

import logging
from fastapi import FastAPI, APIRouter, Query
import my_func as f
import my_func_similarity as s
from typing import List


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()
api_router = APIRouter()


@api_router.get("/business")
def business_endpoint(symbol: str):
    return f.business(symbol)


@api_router.get("/similar")
def similar_endpoint(symbol: str = None, sectors: List[str] = Query(None), limit: int = 5,
                     category: List[str] = Query(None), threshold: float = 0.95):
    # print(sectors)
    similarity_matrix_combined_csv = 'similarity_combined.csv'
    similarity_matrix_sector_csv = 'similarity_sector.csv'

    if symbol is not None:
        return s.similar_companies(similarity_matrix_csv=similarity_matrix_combined_csv, symbol=symbol,
                                   limit=limit, mkcap_category=category, threshold=threshold)
    elif sectors is not None:
        return s.similar_companies(similarity_matrix_csv=similarity_matrix_sector_csv, sectors=sectors,
                                   limit=limit, mkcap_category=category, threshold=threshold)


@api_router.get("/ratios")
def ratios_endpoint(symbols: List[str] = Query(None), market: str = 'US', ratios_type: str = 'overview', convert: str = 'bln'):
    return f.ratios(symbols, market=market, type=ratios_type, convert=convert)


@api_router.get("/capital")
def capital_endpoint(symbol: str, market: str = 'US', capital_type: str = 'overview', convert: str = 'bln'):
    return f.capital(symbol, market=market, type=capital_type, convert=convert)


@api_router.get("/profit")
def profit_endpoint(symbol: str, market: str = 'US', profit_type: str = 'overview', convert: str = 'bln', earnings_time: int = 5):
    return f.profitability(symbol, market=market, type=profit_type, convert=convert, earnings_time=earnings_time)


@api_router.get("/cashflow")
def cashflow_endpoint(symbol: str, market: str = 'US', cashflow_type : str = 'overview', convert: str = 'bln'):
    return f.cashflow(symbol, market=market, type=cashflow_type, convert=convert)


@api_router.get("/one_shot_analysis")
def one_shot_analysis(symbol: str, peers: List[str] = Query([]), market: str = 'US', type : str = 'overview', convert: str = 'bln', earnings_time: int =5):
    return f.one_shot_analysis(symbol, peers=peers, market=market, type=type, convert=convert, earnings_time=earnings_time)


@api_router.get("/news")
def news_endpoint(symbol: str = None, market: str = 'US', limit: int = 5, period: str = '0d', news_day: str = None,
                  tag: str = None, mode: str = 'latest'):
    return f.news(symbol=symbol, market=market, limit=limit, period=period, news_day=news_day, tag=tag, mode=mode)


@api_router.get("/sentiments")
def sentiment_endpoint(symbol: str, market: str = 'US', period: str = '0d', news_day: str = None):
    return f.news_sentiment(symbol=symbol, market=market, period=period, news_day=news_day)


app.include_router(api_router, prefix="/fundamental")









