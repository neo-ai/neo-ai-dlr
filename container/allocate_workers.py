import requests

# (number of workers, number of threads)
grid = {'ml.c5.9xlarge': (4, 8),
        'c5d.9xlarge': (4, 8),
        'c5.9xlarge': (4, 8)}

r = requests.get('http://169.254.169.254/latest/meta-data/instance-type')
assert r.status_code == 200
instance_type = r.text
print('{} {} {}'.format(instance_type, grid[instance_type][0], grid[instance_type][1]))
