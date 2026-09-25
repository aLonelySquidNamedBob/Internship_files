import yaml

with open("config/stars.yaml", "r") as f:
    stars = yaml.safe_load(f)

print(stars["stars"])