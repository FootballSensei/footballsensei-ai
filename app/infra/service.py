from fastapi import APIRouter, HTTPException, Request
from app.infra.config import * 
from joblib import load  
from datetime import timedelta, datetime
import numpy as np 
import praw 

router = APIRouter()
model = load(MODEL_LOCATION)

@router.post("/score", response_model=None)
async def score(r: Request):
    HTP = r.query_params.get('HTP')
    ATP = r.query_params.get('ATP')
    HM1_D = r.query_params.get('HM1_D')
    HM1_L = r.query_params.get('HM1_L')
    HM1_M = r.query_params.get('HM1_M')
    HM1_W = r.query_params.get('HM1_W')
    HM2_D = r.query_params.get('HM2_D')
    HM2_L = r.query_params.get('HM2_L')
    HM2_M = r.query_params.get('HM2_M')
    HM2_W = r.query_params.get('HM2_W')
    HM3_D = r.query_params.get('HM3_D')
    HM3_L = r.query_params.get('HM3_L')
    HM3_M = r.query_params.get('HM3_M')
    HM3_W = r.query_params.get('HM3_W')
    AM1_D = r.query_params.get('AM1_D')
    AM1_L = r.query_params.get('AM1_L')
    AM1_M = r.query_params.get('AM1_M')
    AM1_W = r.query_params.get('AM1_W')
    AM2_D = r.query_params.get('AM2_D')
    AM2_L = r.query_params.get('AM2_L')
    AM2_M = r.query_params.get('AM2_M')
    AM2_W = r.query_params.get('AM2_W')
    AM3_D = r.query_params.get('AM3_D')
    AM3_L = r.query_params.get('AM3_L')
    AM3_M = r.query_params.get('AM3_M')
    AM3_W = r.query_params.get('AM3_W')
    HTGD = r.query_params.get('HTGD')
    ATGD = r.query_params.get('ATGD')
    DiffFormPoints = r.query_params.get('DiffFormPoints')
   
    # partea de clasificare
    x = [[HTP, ATP, HM1_D, HM1_L, HM1_M, HM1_W, HM2_D, HM2_L, HM2_M, HM2_W, HM3_D, HM3_L, HM3_M, HM3_W, AM1_D, AM1_L, AM1_M, AM1_W, AM2_D, AM2_L, AM2_M, AM2_W, AM3_D, AM3_L, AM3_M, AM3_W, HTGD, ATGD, DiffFormPoints]]
    x = np.array(x, dtype=object)
    prediction_prob = model.predict_proba(x)
    print(prediction_prob[0][0])
    return {'score': prediction_prob[0][0].item()} # probabilitatea sa castige echipa de acasa
