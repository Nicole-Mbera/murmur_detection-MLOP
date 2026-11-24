# Deploying on AWS EC2 and running the load tests

The whole system runs from `docker-compose.yml`: several API containers (`api`), an nginx load balancer (`gateway`) on port 80,
the Streamlit UI (`ui`) on port 8501, and an optional Locust container (`locust`, only for load tests).

## 1. Launch the machine (EC2 console)
1. EC2 > Launch instance. Name: `heart-murmur`.
2. Image: Ubuntu Server 24.04 LTS.
3. Instance type: **`m7i-flex.large`** (the console marks it "Free tier eligible"; it has 8 GiB of RAM).
   Do not use `t3.micro`: it has 1 GiB of RAM, and one API container alone needs about 750 MB, so the machine runs out of memory,
   becomes unresponsive and SSH times out. To change the type of an existing instance: stop it, then Actions > Instance settings >
   Change instance type, then start it again.
4. Key pair: create a new one, download the `.pem` file, keep it safe.
5. Network settings > create a security group with:
   - SSH (22): source **My IP**
   - HTTP (80): anywhere
   - Custom TCP 8501: anywhere
6. Storage: 30 GiB gp3. Launch.
7. In Billing, confirm the budget alert you created is active.

## 2. Install Docker on the machine
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<PUBLIC-IP>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit            # log out and back in so the group change applies
ssh -i your-key.pem ubuntu@<PUBLIC-IP>
docker compose version
```

## 3. Get the code and start everything
```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
docker compose up -d --build        # first build takes several minutes
docker compose ps
curl http://localhost/health
```
Then open `http://<PUBLIC-IP>:8501` (UI) and `http://<PUBLIC-IP>/docs` (API docs).

## 4. Change the number of API containers
```bash
scripts/scale.sh 3        # runs 3 API containers behind nginx
```
The script restarts nginx because nginx reads the container addresses only when it starts.
To see requests being spread across containers: `docker compose logs gateway | tail` (each line ends with `upstream=<address>`).

## 5. Load tests (Locust)
```bash
USERS=30 DURATION=120s scripts/run_load_tests.sh
```
This runs the same load against 1, 2 and 3 API containers and writes `results/summary.md`, `results/load_test_summary.png`
and the raw Locust files. Copy them back to your computer with:
```bash
scp -i your-key.pem -r ubuntu@<PUBLIC-IP>:~/<repo>/results ./results_from_aws
```
Notes for interpreting the results:
- The machine has 2 vCPUs, and the Locust container runs on the same machine, so it competes with the API for CPU.
  Expect the gain from 1 to 2 containers to be larger than from 2 to 3. Say so in the README.
- Do not retrain during a load test. Retraining uses the same CPUs.
- Run `scripts/reset_demo.sh` and `docker compose restart api` before a test if you retrained earlier, so the model is v1.

## 6. After changing the code or the model
```bash
git pull
docker compose up -d --build
scripts/reset_demo.sh               # model v1, empty uploads database
docker compose restart api
```
Containers do not update by themselves: the running containers keep the old code until you rebuild.

## 7. Useful commands
```bash
docker compose logs -f api          # API logs
docker stats                        # memory and CPU per container
docker compose down                 # stop everything
scripts/reset_demo.sh               # back to model v1, no uploaded data
```

## 8. Cost and security
- First run `docker compose down`, then stop the instance (EC2 > Instance state > Stop) when you are not using it. Stopping pauses
  compute charges; the disk is still billed. The public IP changes after a stop and start, so update the URL in the README.
- Check the Billing dashboard every day or two.
- The API has no login. Anyone who knows the address can upload data and trigger retraining. Keep the security group tight
  (or stop the instance) outside your demo and the grading period.
- If memory runs out (`docker stats`, `docker compose logs api`), use 2 API containers.
