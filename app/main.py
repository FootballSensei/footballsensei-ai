
from fastapi import FastAPI 
from app.infra.service import router 

app = FastAPI(openapi_url="/api/openapi.json", docs_url="/api/docs")

@app.on_event("startup")
async def startup():
    print("ML microservice started")

@app.on_event("shutdown")
async def shutdown():
    print("ML microservice stopped")

app.include_router(router, prefix="/api", tags=["machine_learning"])
