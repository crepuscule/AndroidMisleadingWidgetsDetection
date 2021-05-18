AKs = ['2eP5WdN647rGCnHXu3SDkVfs','2eP5WdN647rGCnHXu3SDkVfs','5uTq7NlOOKKQ1zPV94Xrxgvg','hICK1UBOZyRCR8Wfte0Zir2l','4ulbLNOWTqImgsuMpiXPwAOy','4LVDODpyxXaf05qcvkqRvpe9','i3yQdP9dzlhBhxqTkr269udk']
SKs = ['iKLHrga4dXQues4VhB8xn4jwdVlEuOBz','Rb5Rx7vunyaWtzyobMog3tCTElEKrLgX','907q3yflGPIRO0CR4rA0HY5ZM3INPGus','0Mu5UQ2T4UT2DGZfoTZtjICWhDV9TkLI','wAABV34lcZR8l7S5V5mXOnGEo2i97van','hN9B9NsGWH8rxj6eQvOcWI6YYUVIhAIt','LFtHXK1VrIaoy1u1clliCxuFVfw27Vg6']
import os,sys

for i in range(len(AKs)):
    print("curl -i -k 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s'" % (AKs[i],SKs[i]))
    os.system("curl -i -k 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s'" % (AKs[i],SKs[i]))
