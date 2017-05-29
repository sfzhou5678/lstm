# encoding:utf-8

import ast

def isValid(field):
    if len(field)==0:
        return False
    if field in ('None','',None):
        return False
    else:
        return True

def str_node(node):

    if isinstance(node, list):
        fields = [str_node(item) for item in node]
        fields=[field for field in fields if isValid(field)]
        if len(fields)>0:
            ans = '%s' % ' '.join('%s' % field for field in fields)
            return ans
        else:
            return ''
    fields=[]
    if isinstance(node, ast.operator):
        # TODO 如 add() mul() 好像还有很多非BinOp的东西？？
        return 'Op'
    elif isinstance(node,ast.Name):
        return 'Name'
    elif isinstance(node,ast.Num):
        return 'Num'
    elif isinstance(node,ast.Str):
        return 'Str'
    elif isinstance(node,ast.arg):
        return 'Arg'
    elif isinstance(node,ast.alias):
        return 'AsName'
    elif isinstance(node,ast.Global):
        fields = [str_node(val) for field, val in ast.iter_fields(node)]
        rv = '%s{%s}' % (node.__class__.__name__, ' '.join('%s' % 'Name' for field in fields))
        return rv
    elif isinstance(node,ast.Bytes):
        return 'Bytes'
    elif isinstance(node,ast.Nonlocal):
        # pass
        return 'Nonlocal'

    elif isinstance(node,ast.Attribute):
        # attr统一表示为AttrName，去掉ctx[不知道ctx代表什么]
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field not in ('attr','ctx')]
        fields.insert(1,'AttrName')
    elif isinstance(node,ast.keyword):
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field not in ('arg')]
        fields.insert(0,'Key')



    elif isinstance(node,ast.ImportFrom):
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field not in ('module','level')]
        fields.insert(0,'ModuleName')
    elif isinstance(node,ast.ExceptHandler):
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field  not in ('name')]

    # 函数、类定义
    elif isinstance(node,ast.ClassDef):
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field not in ('name')]
        fields.insert(0,'ClassName')
    elif isinstance(node,ast.FunctionDef):
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field  in ('args','body','returns')]
        fields.insert(0,'FuncName')
    elif isinstance(node,ast.AsyncFunctionDef):
        fields = [str_node(val) for field, val in ast.iter_fields(node) if field  in ('args','body','returns')]
        fields.insert(0,'AsyncFuncName')

    # 其他默认行为
    elif isinstance(node, ast.AST):
        fields = [str_node(val) for field, val in ast.iter_fields(node)]
        fields=[field for field in fields if isValid(field)]
    else:
        # 特殊情况：什么都不是的node
        return repr(node)

    # 最后在下面统一返回结果
    fields=[field for field in fields if isValid(field)]
    if len(fields)>0:
        rv = '%s{%s}' % (node.__class__.__name__, ' '.join('%s' % field for field in fields))
    else:
        rv=node.__class__.__name__
    return rv

def file_to_type(filepath):
    with open(filepath,'rb') as f:
        code = f.read()
        return (str_node(ast.parse(code)))
    # try:
    #     with open(filepath,'rb') as f:
    #         code = f.read()
    #         return (str_node(ast.parse(code)))
    # except:
    #     return ''

filepath = r'../data/imcomplete_code.py'
# print(file_to_type(filepath))


import py_compile
py_compile.compile(filepath)