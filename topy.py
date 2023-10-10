import nbformat

# ipynb ファイルのパスを指定
ipynb_file_path = 'recog_l2.ipynb'

# ipynb ファイルを読み込む
with open(ipynb_file_path, 'r', encoding='utf-8') as nb_file:
    notebook_content = nbformat.read(nb_file, as_version=4)

# コードセルの内容を抽出
code_cells = [cell['source'] for cell in notebook_content['cells'] if cell['cell_type'] == 'code']

# コードセルの内容を表示または保存
# for i, code_cell in enumerate(code_cells):
#     # print(f"Code Cell {i+1}:\n")
#     print(code_cell)
#     print("\n")
print(ipynb_file_path[:-6])
# コードセルの内容を別のファイルに保存したい場合
with open(f'{ipynb_file_path[:-6]}.py', 'w', encoding='utf-8') as code_file:
    for code_cell in code_cells:
        code_file.write(code_cell + '\n')
