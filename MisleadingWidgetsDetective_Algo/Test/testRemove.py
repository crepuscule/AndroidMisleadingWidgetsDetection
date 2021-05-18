import os 
image_path = '/data/DroidBot_Epoch/raw_data/input_preview/view_ff3a757d5d2ffe81067f7344d1c6acec.png'
md5 = os.popen('md5sum '+image_path)
md5 = md5.read().split('  ')[0]
print(md5)
if md5 == '29fac9bf62c4109a3bf95d022cbd7a53' or md5 == '98eb34e6c7a85ee26d3ac887058b447b' or md5 == '88f7799585f0b0fdcd9478a92a08048a' or md5 == '7781d8d47f1b7e175e4c30d8efa7ed10':
    print('不需要的图片', image_path, ' w, h: ')

