import easyocr,sys

reader = easyocr.Reader(['en'])
input_list = ["view_5c261ffd2ff010ed8595940026e53cfd.png","view_5f4ff95b432cfa28b166053bda27c097.png","view_6a6be836c131f3f7021e87fe47a1d00d.png","view_5ecdf5e7309ea11e4d58e182ad889634.png","view_6a5d46d99a2945eb892655451e2a960a.png"]
#input_list = ["view_3c561a0409fc2ea64de5ea0c80fe0b8e.png","view_3c62f950dfe5a3b38f57532fd5c3d9e5.png","view_3c6810045d2cbb14533f99223d9a379c.png","view_3c9da8fdd9e884323150e16ff2b1d2c4.png","view_3ce096c75172bb12bc2c9bae21de2a26.png","view_3cee3d41cf42711eda8f9304588eadfc.png","view_3d30d1c286f18b956903439dfa84c484.png","view_3d8c722a35db5edb098cf5d9f39ab0d8.png","view_3e88a3a99b8cf04353c80c953dcbb38d.png","view_3eb1c7df278c46060f91932e182cfc18.png","view_3eeb214ece2d1f76bba0db3978e7246e.png","view_3f6adc786346c113b236065fe7f457fc.png","view_4a1030eea495bcd57615f9ab795322a7.png","view_4b1494d7a7c34e92b6e13e00abeab862.png","view_4d53fb6e03e464619029ba18872a8774.png","view_4f3bfd2c2445e93b4e19fc9a90c29c15.png","view_4f7a04d575274c1481dbbeb7dd94dfe1.png","view_4f7ba3a89e52793167224517ef8bdbb7.png" ]
for input_item in input_list:
        print(input_item)
        result = reader.readtext(input_item,detail=1)
        print('>>\n',result,"<<")
