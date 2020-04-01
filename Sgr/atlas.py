import os
import sys
from builtins import input
sys.path.append('../ISM/')
from base_atlas import make_atlas

# parameters
border = 3.
lmin = 46.
lmax = 53.5
nbounds = 3
bfilename = 'Sgr_bound_dump.npy'

make_atlas(lmin,lmax,border,nbounds,bfilename)

# # add boundaries
# x,y = lbd2xy(lmin,0, R0,R0)
# y2 = R0 + 2*x*(y-R0)/x
# ax0.plot([0,2*x],[R0,y2],linestyle='-',color='k')
# x,y = lbd2xy(lmin-border,0, R0,R0)
# y2 = R0 + 2*x*(y-R0)/x
# ax0.plot([0,2*x],[R0,y2],linestyle='--',color='k')
# x,y = lbd2xy(lmax,0, R0,R0)
# y2 = R0 + 2*x*(y-R0)/x
# ax0.plot([0,2*x],[R0,y2],linestyle='-',color='k')
# x,y = lbd2xy(lmax+border,0, R0,R0)
# y2 = R0 + 2*x*(y-R0)/x
# ax0.plot([0,2*x],[R0,y2],linestyle='--',color='k')
# ax1.vlines([lmin,lmax],-157.3, 157.3,linestyle='-',color='k')
# ax1.vlines([lmin-border,lmax+border],-157.3, 157.3,linestyle='--',color='k')
# ax2.vlines([lmin,lmax],-153.17, 153.17,linestyle='-',color='k')
# ax2.vlines([lmin-border,lmax+border],-153.17, 153.17,linestyle='--',color='k')
# ax3.vlines([lmin,lmax],0, 15.,linestyle='-',color='k')
# ax3.vlines([lmin-border,lmax+border],0, 15.,linestyle='--',color='k')
# ax4.vlines([lmin,lmax],0, 15.,linestyle='-',color='k')
# ax4.vlines([lmin-border,lmax+border],0, 15.,linestyle='--',color='k')
#
# # draw boundaries on CO map
# plt.figure(fig1.number)
# # if boundary file already ask to read it
# use_file = False
# if os.path.isfile(bfilename):
#     read = input('Do you want to read boundaries from exisiting file? [yes/no] ')
#     if read == 'yes':
#         use_file = True
#     elif read == 'no':
#         pass
#     else:
#         use_file = True
#         print('Option not recognized. Boundaries will be read from file for safety.\n')
# if use_file == True:
#     bounds = np.load(bfilename)
#     for bound in bounds:
#         ax1.plot(bound[0], bound[1],
#                  linestyle=':', color='k')
# else:
#     print('Zoom and pan as needed, \n'
#           'then press any key to start clicking and draw lines.\n'
#           'You will draw {} separation lines.\n'.format(nbounds))
#     plt.waitforbuttonpress()
#     ld = LineDrawer()
#     for s in range(nbounds):
#         ld.draw_line()
#     # save file
#     bounds = []
#     for line in ld.lines:
#         l = line[0].get_data()[0]
#         v = line[0].get_data()[1]
#         bounds.append([l,v])
#     bounds = np.array(bounds)
#     np.save(bfilename,bounds)
#
# # overlay boundaries to other plots
# # HI
# for bound in bounds:
#     plt.figure(fig2.number)
#     ax2.plot(bound[0],bound[1],
#              linestyle=':',color='k')
#     lon, dist = ld_linear(bound[0],bound[1])
#     for d in dist:
#         if np.all(d == np.isnan):
#             pass
#         else:
#             plt.figure(fig3.number)
#             ax3.plot(lon,d,linestyle=':',color='k')
#             ax4.plot(lon, d, linestyle=':', color='k')
#             x, y = lbd2xy(lon,0,d,R0)
#             plt.figure(fig0.number)
#             ax0.plot(x, y, linestyle=':', color='k')
