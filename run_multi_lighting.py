import os
from multiprocessing import Pool

if __name__ == '__main__':
    gpu_id = 0
    backbone = 'wide_resnet50_2'
    backbone = 'resnet18'
    method_list = ['RD4AD']
    # eyecandies_class = ['CandyCane','ChocolateCookie','ChocolatePraline','Confetto','GummyBear']
    # eyecandies_class = ['HazelnutTruffle','LicoriceSandwich','Lollipop','Marshmallow','PeppermintCandy']
    eyecandies_class = ['CandyCane','ChocolateCookie','ChocolatePraline','Confetto','GummyBear','HazelnutTruffle','LicoriceSandwich','Lollipop','Marshmallow','PeppermintCandy']
#    method_list = ['RD4AD', 'ST', 'PEFM']

    dataset = 'mvtec'

    pool = Pool(processes=9)  # 进程池
    for method in method_list:
        for cls in eyecandies_class:
            sh = f'python train.py --backbone {backbone} ' \
                 f'--gpu_id {gpu_id} ' \
                 f'--method {method} ' \
                 f'--class_name {cls} '

            print(f'exec {sh}')
            # os.system(sh)
            pool.apply_async(os.system, (sh,))

    pool.close()
    pool.join()  # 等待进程结束


