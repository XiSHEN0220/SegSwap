
import argparse
import os
import pickle
import numpy as np 

import PIL.Image as Image 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument(
    '--input', type=str, help='image directory')

parser.add_argument(
    '--img-size', type=int, nargs='+', default=[480, 480], help='image size')


args = parser.parse_args()
def plotCorres(IA, IB, xA, yA, xB, yB, mask, img_size, saveFig = 'toto.jpg') : 
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    nbPoint = len(xA)
    
    ax1.imshow(np.array(IA))
    ax1.axis('off')
    ax2.imshow(np.array(IB))
    ax2.axis('off')
    
    
    for i in range(nbPoint) : 
        if mask[i] > 0.5: 
            xyA = (int(xA[i] * img_size[0]), int(yA[i] * img_size[1]))
            xyB = (int(xB[i] * img_size[0]), int(yB[i] * img_size[1]))

            con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color='green', linewidth = 0.5)
            ax2.add_artist(con)

            ax1.plot(xyA[0],xyA[1],'ro',markersize=0.5)
            ax2.plot(xyB[0],xyB[1],'ro',markersize=0.5)
    plt.savefig(saveFig, bbox_inches='tight')
    plt.close(fig)
    

stride_net = 16
nb_feat_w = args.img_size[0] / stride_net
nb_feat_h = args.img_size[1] / stride_net

x = ( np.arange(nb_feat_w) + 0.5 ) / nb_feat_w
y = ( np.arange(nb_feat_h) + 0.5 ) / nb_feat_h
grid_x, grid_y = np.meshgrid(x, y)



for file in os.listdir(args.input) : 
    if 'html' in file or 'css' in file : 
        cmd = 'rm {}'.format(os.path.join(args.input, file))
        os.system(cmd)
        
outHtml = os.path.join(args.input, 'vis.html')   

nb_pair = [1 for img in os.listdir(args.input) if 'npy' in img]
nb_pair = np.sum(nb_pair) // 2
f = open(outHtml, 'w')
f.write('<html>\n')
f.write('<head>\n')
f.write('\t<meta name=\"keywords\" content= \"Visual Result\" />  <meta charset=\"utf-8\" />\n')
f.write('\t<meta name=\"robots\" content=\"index, follow\" />\n')
f.write('\t<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\" />\n')
f.write('\t<meta http-equiv=\"expires\" content=\"0\" />\n')
f.write('\t<meta name=\"description\" content= \"Project page of style.css\" />\n')
f.write('\t<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" media=\"screen\" />\n')
f.write('\t<link rel=\"shortcut icon\" href=\"favicon.ico\" />\n')
f.write('</head>\n')
f.write('<body>\n')
f.write('<div id=\"website\">\n')
f.write('<table>\n')
caption = '\t<caption>Annotations</caption>\n'
f.write(caption)

f.write('\t<tr>\n')
f.write('\t\t <td>Idx </td>')
f.write('\t\t <td>Pos 1</td>')
f.write('\t\t <td>Pos 1S</td>')

f.write('\t\t <td>Pos 2</td>')
f.write('\t\t <td>Pos 2S</td>')

f.write('\t\t <td>Mask 1</td>')
f.write('\t\t <td>Mask 1S</td>')

f.write('\t\t <td>Mask 2</td>')
f.write('\t\t <td>Mask 2S</td>')

f.write('\t\t <td>Corr1-->2</td>')
f.write('\t\t <td>Corr1-->2S</td>')
f.write('\t\t <td>Corr1S-->2</td>')
f.write('\t\t <td>Corr1S-->2S</td>')

f.write('\t\t <td>Corr2-->1</td>')
f.write('\t\t <td>Corr2-->1S</td>')
f.write('\t\t <td>Corr2S-->1</td>')
f.write('\t\t <td>Corr2S-->1S</td>')




f.write('\t</tr>\n')

for i in tqdm(range(nb_pair)) : 
    
    f.write('\t<tr>\n')
    
    msg = '\t\t<td>{}</td>\n'.format(i+1)
    f.write(msg) 
    
    
    imga = '{:d}_a.jpg'.format(i)
    imgb = '{:d}_b.jpg'.format(i)
    
    imgas = '{:d}_as.jpg'.format(i)
    imgbs = '{:d}_bs.jpg'.format(i)
    
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imga, imga)
    f.write(msg) 
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imgas, imgas)
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imgb, imgb)
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imgbs, imgbs)
    f.write(msg) 
    
    Ia = np.array(Image.open(os.path.join(args.input, imga)).convert('RGBA'))
    Ib = np.array(Image.open(os.path.join(args.input, imgb)).convert('RGBA'))
    Ias = np.array(Image.open(os.path.join(args.input, imgas)).convert('RGBA'))
    Ibs = np.array(Image.open(os.path.join(args.input, imgbs)).convert('RGBA'))
    
    infoa = np.load(os.path.join(args.input, imga.replace('.jpg', '.npy')))
    infob = np.load(os.path.join(args.input, imgb.replace('.jpg', '.npy')))
    
    maska = np.array(Image.fromarray((np.clip(infoa[2] * 255, a_min=70, a_max=255)).astype(np.uint8)).resize((args.img_size[0], args.img_size[1]), resample=2))
    maskb = np.array(Image.fromarray((np.clip(infob[2] * 255, a_min=70, a_max=255)).astype(np.uint8)).resize((args.img_size[0], args.img_size[1]), resample=2))
    
    
    Ia[:,:,3] = maska
    Ib[:,:,3] = maskb
    Ias[:,:,3] = maska
    Ibs[:,:,3] = maskb
    
    Image.fromarray(Ia).save( os.path.join(args.input, imga.replace('.jpg', '.png')))
    Image.fromarray(Ib).save( os.path.join(args.input, imgb.replace('.jpg', '.png')))
    Image.fromarray(Ias).save( os.path.join(args.input, imgas.replace('.jpg', '.png')))
    Image.fromarray(Ibs).save( os.path.join(args.input, imgbs.replace('.jpg', '.png')))
        
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imga.replace('.jpg', '.png'), imga.replace('.jpg', '.png'))
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imgas.replace('.jpg', '.png'), imgas.replace('.jpg', '.png'))
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imgb.replace('.jpg', '.png'), imgb.replace('.jpg', '.png'))
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format(imgbs.replace('.jpg', '.png'), imgbs.replace('.jpg', '.png'))
    f.write(msg) 
    
    xa, ya = infoa[0], infoa[1]
    xb, yb = infob[0], infob[1]
    out_corrab = os.path.join(args.input, '{:d}_corrab.jpg'.format(i))
    out_corrabs = os.path.join(args.input, '{:d}_corrabs.jpg'.format(i))
    out_corrasb = os.path.join(args.input, '{:d}_corrasb.jpg'.format(i))
    out_corrasbs = os.path.join(args.input, '{:d}_corrasbs.jpg'.format(i))
    
    out_corrba = os.path.join(args.input, '{:d}_corrba.jpg'.format(i))
    out_corrbas = os.path.join(args.input, '{:d}_corrbas.jpg'.format(i))
    out_corrbsa = os.path.join(args.input, '{:d}_corrbsa.jpg'.format(i))
    out_corrbsas = os.path.join(args.input, '{:d}_corrbsas.jpg'.format(i))
    
    Ia = Image.open(os.path.join(args.input, imga)).convert('RGB')
    Ib = Image.open(os.path.join(args.input, imgb)).convert('RGB')
    Ias = Image.open(os.path.join(args.input, imgas)).convert('RGB')
    Ibs = Image.open(os.path.join(args.input, imgbs)).convert('RGB')
    
    plotCorres(Ia, Ib, grid_x.reshape(-1), grid_y.reshape(-1), xa.reshape(-1), ya.reshape(-1), infoa[2].reshape(-1), args.img_size, out_corrab)
    plotCorres(Ia, Ibs, grid_x.reshape(-1), grid_y.reshape(-1), xa.reshape(-1), ya.reshape(-1), infoa[2].reshape(-1), args.img_size, out_corrabs)
    plotCorres(Ias, Ib, grid_x.reshape(-1), grid_y.reshape(-1), xa.reshape(-1), ya.reshape(-1), infoa[2].reshape(-1), args.img_size, out_corrasb)
    plotCorres(Ias, Ibs, grid_x.reshape(-1), grid_y.reshape(-1), xa.reshape(-1), ya.reshape(-1), infoa[2].reshape(-1), args.img_size, out_corrasbs)
    
    plotCorres(Ib, Ia, grid_x.reshape(-1), grid_y.reshape(-1), xb.reshape(-1), yb.reshape(-1), infob[2].reshape(-1), args.img_size, out_corrba)
    plotCorres(Ib, Ias, grid_x.reshape(-1), grid_y.reshape(-1), xb.reshape(-1), yb.reshape(-1), infob[2].reshape(-1), args.img_size, out_corrbas)
    plotCorres(Ibs, Ia, grid_x.reshape(-1), grid_y.reshape(-1), xb.reshape(-1), yb.reshape(-1), infob[2].reshape(-1), args.img_size, out_corrbsa)
    plotCorres(Ibs, Ias, grid_x.reshape(-1), grid_y.reshape(-1), xb.reshape(-1), yb.reshape(-1), infob[2].reshape(-1), args.img_size, out_corrbsas)
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrab.jpg'.format(i), '{:d}_corrab.jpg'.format(i))
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrabs.jpg'.format(i), '{:d}_corrabs.jpg'.format(i))
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrasb.jpg'.format(i), '{:d}_corrasb.jpg'.format(i))
    f.write(msg) 
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrasbs.jpg'.format(i), '{:d}_corrasbs.jpg'.format(i))
    f.write(msg) 
    
    
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrba.jpg'.format(i), '{:d}_corrba.jpg'.format(i))
    f.write(msg)
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrbas.jpg'.format(i), '{:d}_corrbas.jpg'.format(i))
    f.write(msg)
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrbsa.jpg'.format(i), '{:d}_corrbsa.jpg'.format(i))
    f.write(msg)
    
    msg = '\t\t<td> <a href=\"{}\" title="ImageName"> <img  src=\"{}\"/></a></td>\n'.format('{:d}_corrbsas.jpg'.format(i), '{:d}_corrbsas.jpg'.format(i))
    f.write(msg)
    
    f.write('\t\t</tr>\n')
    
    
f.write('</table>\n')
    
f.write('</center>\n</div>\n </body>\n</html>\n')

f.close()    
    
cmd = 'cp style.css {}'.format(args.input)
os.system(cmd)    




    

