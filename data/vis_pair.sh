python generate_1obj.py --out-dir vis10_1obj/ --end-idx 10
python file2web1obj.py --input vis10_1obj

python generate_2obj.py --out-dir vis10_2obj/ --end-idx 10
python file2web2obj.py --input vis10_2obj

firefox vis10_1obj/vis.html
firefox vis10_2obj/vis.html




