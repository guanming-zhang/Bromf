import subprocess

# #cancelling multipe jobs
# job_id_end = 42348005
# job_id_start = 42348008
# for job_id in range(job_id_start,job_id_end):
#     subprocess.run(["scancel",str(job_id)])

#cancelling 1 job
job_id = 54246610
subprocess.run(["scancel",str(job_id)])

