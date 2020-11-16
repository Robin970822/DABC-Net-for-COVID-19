mkdir -p 2020035365
cd 2020035365
gdown --id 1g3Hi_utRCV_c2J6B-QA0sMEM4x8FT_Cx
unzip 2020035365.zip
cd ..

mkdir -p 2020035021
cd 2020035021
gdown --id 1u3womSiXn0cmlkMQLikUAcHj28lud4_t
unzip 2020035021.zip
cd ..

mkdir -p weight
cd weight
gdown --id 1Me-9TqTUMse1lbhGfxj2-CSSipq11nSH
gdown --id 1_79uQZpZYnik2SbNPm-b4GNnTt96dPOx
gdown --id 1BhBU6OxfIpMJsYfD5zkHMhhqJXZpmAg5

gdown --id 1S9AfoItRqo7pHhSEVsoDa4wGm_h_1Afc
unzip model.zip
cd ..

mkdir -p meta
cd meta
gdown --id 1K2a6v708Eo7G64vq0YSVw_uinPcbipLH
gdown --id 1j297o01OI2jEOjBzTVQyzmmFSJOW-kcW
cd ..

gdown --id 1hIJyil3ME7JWJLUOYkX7KlWsJo9NRP5J
unzip output.zip

export PYTHONPATH=$PYTHONPATH:$(pwd)
