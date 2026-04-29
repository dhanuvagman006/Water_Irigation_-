import httpx
import json

async def test_nasa_api():
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PS&community=RE&longitude=74.8560&latitude=12.9141&start=20260427&end=20260428&format=JSON"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        data = response.json()
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_nasa_api())
