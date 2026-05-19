import os
import hashlib
from pathlib import Path
from typing import List, Tuple


def get_all_files(root_dir: str) -> dict:
    """
    递归获取目录下所有文件及其相对路径
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        字典，key为相对路径，value为绝对路径
    """
    root_path = Path(root_dir).resolve()
    files = {}
    
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_path = abs_path.relative_to(root_path)
            files[str(rel_path)] = str(abs_path)
    
    return files


def files_are_equal(file1: str, file2: str, chunk_size: int = 8192) -> bool:
    """
    高效比较两个文件的内容是否一致
    
    先比较文件大小，再逐块计算MD5哈希，避免一次性读入大文件
    
    Args:
        file1: 第一个文件路径
        file2: 第二个文件路径
        chunk_size: 读取文件的块大小
        
    Returns:
        True表示内容一致，False表示不一致
    """
    # 先比较文件大小
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False
    
    # 计算MD5哈希
    hasher1 = hashlib.md5()
    hasher2 = hashlib.md5()
    
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            chunk1 = f1.read(chunk_size)
            chunk2 = f2.read(chunk_size)
            
            if not chunk1 and not chunk2:
                break
                
            hasher1.update(chunk1)
            hasher2.update(chunk2)
    
    return hasher1.hexdigest() == hasher2.hexdigest()


def compare_directories(path1: str, path2: str) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    对比两个目录中相对路径相同的文件内容
    
    Args:
        path1: 第一个目录路径
        path2: 第二个目录路径
        
    Returns:
        元组包含三个元素：
        1. 内容不一致的文件对列表 [(rel_path, path1_file, path2_file), ...]
        2. 只在path1中存在的文件列表
        3. 只在path2中存在的文件列表
    """
    # 验证输入路径
    if not os.path.isdir(path1):
        raise NotADirectoryError(f"路径不存在或不是目录: {path1}")
    if not os.path.isdir(path2):
        raise NotADirectoryError(f"路径不存在或不是目录: {path2}")
    
    # 获取两个目录下的所有文件
    files1 = get_all_files(path1)
    files2 = get_all_files(path2)
    
    # 找出共同存在的文件
    common_files = set(files1.keys()) & set(files2.keys())
    
    # 找出只在一个目录中存在的文件
    only_in_path1 = sorted(set(files1.keys()) - common_files)
    only_in_path2 = sorted(set(files2.keys()) - common_files)
    
    # 对比共同文件的内容
    different_files = []
    for rel_path in sorted(common_files):
        file1 = files1[rel_path]
        file2 = files2[rel_path]
        
        try:
            if not files_are_equal(file1, file2):
                different_files.append((rel_path, file1, file2))
        except Exception as e:
            print(f"警告：无法比较文件 {rel_path}: {str(e)}")
    
    return different_files, only_in_path1, only_in_path2


def main():
    """命令行使用示例"""
    import sys
    
    if len(sys.argv) != 3:
        print("使用方法: python dir_compare.py <目录1> <目录2>")
        sys.exit(1)
    
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    
    try:
        different, only1, only2 = compare_directories(path1, path2)
        
        print(f"=== 目录对比结果 ===")
        print(f"目录1: {path1}")
        print(f"目录2: {path2}")
        print()
        
        if different:
            print(f"❌ 内容不一致的文件 ({len(different)}):")
            for rel_path, file1, file2 in different:
                print(f"  - {rel_path}")
            print()
        else:
            print("✅ 所有共同文件内容一致")
            print()
        
        if only1:
            print(f"📁 只在目录1中存在的文件 ({len(only1)}):")
            for file in only1:
                print(f"  - {file}")
            print()
        
        if only2:
            print(f"📁 只在目录2中存在的文件 ({len(only2)}):")
            for file in only2:
                print(f"  - {file}")
            print()
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()