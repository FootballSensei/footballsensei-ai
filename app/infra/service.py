from fastapi import APIRouter, HTTPException, Request
from app.infra.config import * 
from app.ml.sentiment.sentiment import SentimentAnalysis
from app.infra.secrets import * 
from joblib import load  
from datetime import timedelta, datetime
import numpy as np 
import praw 

router = APIRouter()
model = load(MODEL_LOCATION)
label_encoder = None 
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

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


@router.get("/sentiment", response_model=None)
async def sentiment(r: Request):
    json_body = await r.json()
    team1_name = json_body.get('team1_name')
    team2_name = json_body.get('team2_name')
    team1_players = json_body.get('team1_players')
    team2_players = json_body.get('team2_players')
    match_date = json_body.get('match_date')
    parsed_match_date = datetime.strptime(match_date, '%Y-%m-%d')
    date_since = parsed_match_date - timedelta(days=7)
    date_until = parsed_match_date - timedelta(days=2)
    sentim = SentimentAnalysis(team1_name, team2_name, team1_players, team2_players, date_since, date_until, reddit, 'basic')
    return {'team1': sentim.get_team_sentiment(1), 'team2': sentim.get_team_sentiment(2)}



@router.get("/contextual", response_model=None)
async def contextual(r: Request):
    json_body = await r.json()
    team1_name = json_body.get('team1_name')
    team2_name = json_body.get('team2_name')
    team1_players = json_body.get('team1_players')
    team2_players = json_body.get('team2_players')
    match_date = json_body.get('match_date')
    parsed_match_date = datetime.strptime(match_date, '%Y-%m-%d')
    date_since = parsed_match_date - timedelta(days=7)
    date_until = parsed_match_date - timedelta(days=2)
    sentim = SentimentAnalysis(team1_name, team2_name, team1_players, team2_players, date_since, date_until, reddit, 'advanced')

    raspuns = {}
    teams = ['team1', 'team2']
    ctx = ['Positive', 'Negative']
    for team in teams:
        sub_dict = {}
        for c in ctx:
            sub_dict[c] = sentim.get_team_biased_sentiment(team, c)
        raspuns[team] = sub_dict
    return raspuns