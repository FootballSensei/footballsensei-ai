
from fastapi.testclient import TestClient
import unittest

  
import main

client = TestClient(main.app)

class TestApp(unittest.TestCase):
    def test_successful_request(self):
        response = client.post("/api/score?HTP=-1.3694294447306334&ATP=-1.3813688554778125&HM1_D=0&HM1_L=0&HM1_M=1&HM1_W=0&HM2_D=0&HM2_L=0&HM2_M=1&HM2_W=0&HM3_D=0&HM3_L=0&HM3_M=1&HM3_W=0&AM1_D=0&AM1_L=0&AM1_M=1&AM1_W=0&AM2_D=0&AM2_L=0&AM2_M=1&AM2_W=0&AM3_D=0&AM3_L=0&AM3_M=1&AM3_W=0&HTGD=0.01023228020913664&ATGD=-0.010144663647033121&DiffFormPoints=0.06328516755080839")
        assert response.status_code == 200

        decoded = response.json()
        assert 'score' in decoded
        assert 0 <= decoded['score'] <= 1

    def test_missing_parameters(self):
        response = client.post("/api/score")
        assert response.status_code == 400
        assert response.json()['detail'] == "Bad request"

        params = {
            "HTP":"-1.3694294447306334",
            "ATP":"-1.3813688554778125",
            "HM1_D":"0",
            "HM1_L":"0",
            "HM1_M":"1",
            "HM1_W":"0",
            "HM2_D":"0",
            "HM2_L":"0",
            "HM2_M":"1",
            "HM2_W":"0",
            "HM3_D":"0",
            "HM3_L":"0",
            "HM3_M":"1",
            "HM3_W":"0",
            "AM1_D":"0",
            "AM1_L":"0",
            "AM1_M":"1",
            "AM1_W":"0",
            "AM2_D":"0",
            "AM2_L":"0",
            "AM2_M":"1",
            "AM2_W":"0",
            "AM3_D":"0",
            "AM3_L":"0",
            "AM3_M":"1",
            "AM3_W":"0",
            "HTGD":"0.01023228020913664",
            "ATGD":"-0.010144663647033121",
            "DiffFormPoints":"0.06328516755080839"
        }

        for key in params:
            # get all params except key 
            p = ""
            for k, v in params.items():
                if k != key:
                    p += f"&{k}={v}"
                
            p = p[1:]
            response = client.post(f"/api/score?{p}")
            assert response.status_code == 400

