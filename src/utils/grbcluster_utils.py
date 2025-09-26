import subprocess
import json
import os
import pdb
import pandas as pd


def abort_existing_grbcluster_jobs(solver_api_tokens_path, grb_user='stopfer'):
    with open(solver_api_tokens_path, "r") as json_file:
        solver_api_tokens_data = json.load(json_file)
    
    server = solver_api_tokens_data['GUROBI_COMPUTE_SERVER_NAME']
    password = solver_api_tokens_data['GUROBI_COMPUTE_SERVER_ADMINPASS']
    
    # Login
    login_cmd = [
        "grbcluster", "login",
        f"--server={server}",
        f"--password={password}"
    ]
    subprocess.run(login_cmd, check=True)
    
    # Jobs abrufen
    list_cmd = [
        "grbcluster", "jobs"
    ]
    result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
    
    jobs = result.stdout
    lines = jobs.strip().splitlines()
    
    # Header und Daten trennen
    header_line = lines[0]
    data_lines = lines[1:]
    
    # Spaltenstartpositionen ermitteln (Beginn jedes Wortes im Header)
    positions = [i for i in range(1, len(header_line)) if header_line[i] != ' ' and header_line[i - 1] == ' ']
    positions = [0] + positions  # Sicherstellen, dass erste Spalte bei 0 beginnt
    
    # Spaltennamen extrahieren (von jeder Startposition bis zur nächsten)
    columns = [header_line[positions[i]:positions[i + 1]].strip()
               for i in range(len(positions) - 1)]
    columns.append(header_line[positions[-1]:].strip())
    
    # Datenzeilen anhand der Startpositionen parsen
    records = []
    for line in data_lines:
        fields = [line[positions[i]:positions[i + 1]].strip()
                  for i in range(len(positions) - 1)]
        fields.append(line[positions[-1]:].strip())
        if any(fields):
            records.append(fields)
    
    # DataFrame erzeugen
    job_df = pd.DataFrame(records, columns=columns)
    if len(job_df) > 0:
        print(f"current jobs: \n {job_df} \n")    
    
        # Alle Jobs von grb-user abbrechen
        for row_idx, job in job_df.iterrows():
            job_id = job['JOBID']
            user = job['USER']
            if user == grb_user:
                subprocess.run([
                    "grbcluster", "job", "abort", job_id,
                    ], check=True)
                print(f"ended {server} job: {job_id} by user: {user}")
    
    else:
        print(f"there are currently no jobs running on {server}")
    return job_df

if __name__ == "__main__":
    solver_api_tokens_path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config'), 'config_files'), 'solver_api_tokens.json')
    jobs_df = abort_existing_grbcluster_jobs(solver_api_tokens_path)
