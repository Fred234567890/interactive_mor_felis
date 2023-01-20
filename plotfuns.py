# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:11:43 2022

@author: Fred

Library for standardized plot functions 
"""

import utility as ut

import os
import sys
import numpy as np
import numpy.linalg as nlina
import scipy 
import numpy.matlib
import warnings


import scipy.interpolate as sint
import kaleido; print(kaleido.__version__)
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# import pickle


###############################################################################
### Constants    
def colorList(colIndex=None,opac=None,name='default'):        
    if name=='blue':
        return colorList(colIndex=0)
    elif name=='orange':
        return colorList(colIndex=1)
    elif name=='green':
        return colorList(colIndex=2)
    elif name=='red':
        return colorList(colIndex=3)
    elif name=='purple':
        return colorList(colIndex=4)
    elif name=='black':
        return 'black'
        
    # colIndex+=6
    if name=='default':
        cols=list(px.colors.DEFAULT_PLOTLY_COLORS)
        colsRgb=plotly.colors.convert_colors_to_same_type(cols, colortype='rgb')[0]
        
        if opac!=None:
            for i in range(len(colsRgb)):
                colsRgb[i]='rgba'+colsRgb[i][3:-1]+','+str(opac)+')'
                
        if colIndex!=None:
            colIndex+=0
            colsRgb=colsRgb[colIndex%len(colsRgb)]
            
    elif name=='rising':
        colsRgb=px.colors.sample_colorscale('turbo',colIndex)[0]
        
    else:
        raise Exception('unknown color list')
           
    return colsRgb 


###############################################################################
### Low level funs    
def plotVariance(fig,fAxis,yMean,deviation,fillArgs,lim=(-np.inf,np.inf)):
    #mean,deviation as absolute values
    #based on https://plotly.com/python/line-charts/ 4.3.22
    yMin=np.clip(yMean-deviation,lim[0],lim[1])
    yMax=yMean+deviation
    
    #variance interval
    curve=go.Scatter(
    x=ut.append(fAxis,fAxis[::-1]),
    y=ut.append(yMin,yMax[::-1]),
    fill='toself',
    fillcolor=fillArgs['color'],  #'rgba(0,100,80,0.2)',
    line_color='rgba(0,0,0,0)',
    showlegend=fillArgs['showLegend'],
    name=fillArgs['name']
    )
    fig.add_trace(curve)
    
    
def plotLine(fig,fAxis,y,lineArgs=dict(),lim=(-np.inf,np.inf)):
    if not 'color' in lineArgs.keys():lineArgs['color']=0
    if not  'dash' in lineArgs.keys():lineArgs['dash']='solid'
    if not  'showLegend' in lineArgs.keys():lineArgs['showLegend']=True
    if not  'name' in lineArgs.keys():lineArgs['name']=''
        
    if type(lineArgs['color'])==int:lineArgs['color']=colorList(lineArgs['color'])
        
    y=np.clip(y,lim[0],lim[1])
    curve=go.Scatter(
    x=fAxis,
    y=y,
    line_color=lineArgs['color'],
    line_dash=lineArgs['dash'],
    showlegend=lineArgs['showLegend'],
    name=lineArgs['name']
    )    
    fig.add_trace(curve)
    
    
def plotCloud(fig,x,y,cloudArgs,lim=(-np.inf,np.inf)):
    y=np.clip(y,lim[0],lim[1])
    cloud=go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker_color=cloudArgs['color'],
    showlegend=cloudArgs['showLegend'],
    name=cloudArgs['name'],
    marker_size=10
    )    
    fig.add_trace(cloud)
    
    
###############################################################################
### Mid level funs
def initPlot(title='',legendTitle='',logX=True,logY=True,xName=None,yName=None,xRange=None,yRange=None):
    logx=logy='linear'    
    if logX:
        logx='log'
    if logY:
        logy='log'
        
        
    fig=go.Figure()    
    fig.update_layout(title=title)
    
    if legendTitle!=None:
        if len(legendTitle)>0:
            fig.update_layout(legend_title_text=legendTitle)
    
    fig.update_xaxes(type=logx)  #{'linear','log'}
    fig.update_yaxes(type=logy)
    
    fig.update_xaxes(title_text=xName)
    fig.update_yaxes(title_text=yName)
    
    if xRange!=None:
        if not logX:
            fig.update_xaxes(range=xRange)
        else:
            fig.update_xaxes(range=np.log10(xRange))
            
    if yRange!=None:
        if not logY:
            fig.update_yaxes(range=yRange)
        else:
            fig.update_yaxes(range=np.log10(yRange))            
    return fig


def plotVariances(fig,fAxis,YMean,deviations,plotInd=-1,lim=(-np.inf,np.inf),plotMean=True,plotVar=True,staticColor=False,showLegend=False):
    fillArgs=dict()
    fillArgs['Opacity']=0.2
    fillArgs['showLegend']=False
    
    lineArgs=dict()
    lineArgs['dash']='dash'
    lineArgs['showLegend']=showLegend
    
    if lim==None:
        lim=(np.min(YMean)*0.5,np.inf)
    
    if  plotInd==-1:
        plotInd=range(np.shape(YMean)[0])
        
    #create curves
    sameColor=1
    for i in range(len(plotInd)):
        if plotMean:
            curveName='mean'
            if staticColor:
                lineArgs['color']=colorList(int(plotInd[i]/sameColor))
            else:            
                lineArgs['color']=colorList(int(i/sameColor))
            if type(curveName)==str:
                lineArgs['name']=curveName+str(i)
            elif type(curveName)==list or type(curveName)==np.ndarray:
                lineArgs['name']=curveName[i]
            else:
                lineArgs['name']=None
                warnings.warn('unknown type of curveName')
            plotLine(fig,fAxis,YMean[plotInd[i]],lineArgs,lim=lim)
            
        if plotVar:
            curveName='var'
            if staticColor:
                fillArgs['color']=colorList(int(plotInd[i]/sameColor),opac=0.3)
            else:            
                fillArgs['color']=colorList(int(i/sameColor),opac=0.3)
            if type(curveName)==str:
                fillArgs['name']=curveName+str(i)
            elif type(curveName)==list or type(curveName)==np.ndarray:
                fillArgs['name']=curveName[i]
            else:
                fillArgs['name']=None
                warnings.warn('unknown type of curveName')
            plotVariance(fig,fAxis,YMean[plotInd[i]],deviations[i],fillArgs,lim=lim) 
    fig.update_traces(mode='lines')
        

def plotLines(fig,fAxis,Y,plotInd=-1,curveName='',dash='solid',showLegend=True,lim=(-np.inf,np.inf),sameColor=1,staticColor=False):
    #configure fig
    Y=np.real(Y)
        
        
    if  plotInd==-1:
        plotInd=range(np.shape(Y)[0])
    # plotInd=np.array(plotInd)
    
    lineArgs=dict()
    lineArgs['dash']=dash
    lineArgs['showLegend']=showLegend
    
    if lim==None:
        lim=(np.min(Y)*0.5,np.inf)
    
    for i in range(len(plotInd)):
        if staticColor:
            lineArgs['color']=colorList(int(plotInd[i]/sameColor))
        else:            
            lineArgs['color']=colorList(int(i/sameColor))
        if type(curveName)==str:
            lineArgs['name']=curveName+str(i)
        elif type(curveName)==list or type(curveName)==np.ndarray:
            lineArgs['name']=curveName[i]
        else:
            lineArgs['name']=None
            warnings.warn('unknown type of curveName')
            
        plotLine(fig,fAxis,Y[plotInd[i]],lineArgs,lim=lim)
        

def plotClouds(fig,X,Y,plotInd=-1,cloudName='',showLegend=True,lim=(-np.inf,np.inf),colorlistName='default'):
    
    if  plotInd==-1:
        plotInd=range(np.shape(Y)[0])
    
    
    cloudArgs=dict()
    cloudArgs['showLegend']=showLegend
    
    for i in plotInd:
        if colorlistName=='default':            
            cloudArgs['color']=colorList(i)  
        else:
            if len(plotInd)>1:
                cloudArgs['color']=colorList(0.1+0.8*i/len(plotInd),name='rising')
            else:
                cloudArgs['color']=colorList(0.5,name='rising')
            
        if type(cloudName)==str:
            cloudArgs['name']=cloudName+str(i)
        elif type(cloudName)==list or type(cloudName)==np.ndarray:
            cloudArgs['name']=cloudName[i]
        else:
            cloudArgs['name']=None
            warnings.warn('unknown type of cloudName')
            
        plotCloud(fig,X[i],Y[i],cloudArgs,lim=lim)
        
    
###############################################################################
### High level funs
    
def plotFRFs(fAxis,Y,plotInd=-1,logX=True,logY=True,title=None,ident='base',sameColor=1,xRange=None,yRange=None,curveName='',show=True,ret=False):
    """
    Parameters
    ----------
    fAxis : 1D-Array of float
    Y : 2D-Array of float
    plotInd : 1D-Array of float, optional
        Curves wich will be plotted. -1 corresponds to all. The default is -1
    drawPlot : Boolean, optional
        wether plot should be shown. The default is True
    log_x : Boolean, optional
        The default is True.
    log_y : Boolean, optional
        The default is True.
    title : String, optional
        The default is ''.

    Returns
    -------
    Fig if requested.
    """
    if (show or ret) is False:
        return
    xName,yName=('x','y')#labelNamesFRF(ident) 
    fig=initPlot(title=title,legendTitle=None,logX=logX,logY=logY,xName=xName,yName=yName,xRange=xRange,yRange=yRange)
    plotLines(fig,fAxis,Y,plotInd=plotInd,curveName=curveName,dash='solid',showLegend=True,sameColor=sameColor)
    if show:
        showPlot(fig)
    if ret:
        return fig


###############################################################################
###Rendering
def numTracesWithName(fig):
    num=0
    for trace in fig._data:
        if 'name' in trace.keys():
            if trace['name']!=None:
                if len(trace['name'])>0:
                    num+=1
    return num
    
def showPlot(fig,renderer='browser',show=True):
    if show:
        pio.renderers.default=renderer #{'svg','browser'}
        fig.show()

    
def savePlot(fig,fileName,path=None):
    if path==None: raise Exception('path has to be assigned in this version')   
    pio.write_json(fig,path+fileName+'.json')
    
    
def loadPlot(fileName,path=None):
    if path==None: raise Exception('path has to be assigned in this version')
    return pio.read_json(path+fileName+'.json')
    
def exportPlot(fig,fileName,size,plotType=None,path=None,opts=dict()):
    #configure opts
    if not 'gridShow' in opts.keys(): opts['gridShow']=True    
    if not 'legendShow' in opts.keys(): opts['legendShow']=False
    if not 'legendPos' in opts.keys(): opts['legendPos']='topRight'
    if not 'xTick' in opts.keys(): opts['xTick']=None
    if not 'yTick' in opts.keys(): opts['yTick']=None
    if not 'tickvalsX' in opts.keys(): opts['tickvalsX']=None
    if not 'ticktextX' in opts.keys(): opts['ticktextX']=None
    if not 'tickvalsY' in opts.keys(): opts['tickvalsY']=None
    if not 'ticktextY' in opts.keys(): opts['ticktextY']=None
    if not 'xRange' in opts.keys(): opts['xRange']=None
    if not 'yRange' in opts.keys(): opts['yRange']=None
    if not 'xFormat' in opts.keys(): opts['xFormat']='~s' #plotly defaul: ''
    if not 'yFormat' in opts.keys(): opts['yFormat']='~s' #plotly defaul: ''
    if not 'xSuffix' in opts.keys(): opts['xSuffix']=None
    if not 'ySuffix' in opts.keys(): opts['ySuffix']=None
    
    
    #setup export path
    if path==None: raise Exception('path has to be assigned in this version')
    if '\\' in fileName: raise Exception('fileName cant be a Path')    
    filePathName=path+fileName
        
    # #save unconfigured fig to file
    # savePlot(fig,fileName,path=path)
    
    #Base Values
    fontCol='black'
    gridCol='rgb(210,210,210)'
    gridThickness=2
    fontSmall=20*1.3
    fontMedium=26*1.15
    fontLarge=34*1.3
    line_width=2.64*1.15
    dot_size=10*1.15
    
    marginWidthR=1
    marginWidthT=1
 
    if 'title' in fig._layout['xaxis'].keys():
        marginWidthB=70
    else:
        marginWidthB=40    
       
    if 'title' in fig._layout['yaxis'].keys():
        marginWidthL=120
    else:
        marginWidthL=90
        
    spaceLegend=0.01
    
    if size.lower()=='full':
        height=480
        width=1280  
        fontSmall=fontSmall
        fontMedium=fontMedium
        fontLarge=fontLarge
        
        marginWidthR=marginWidthL
        marginWidthT=marginWidthT
        marginWidthL=marginWidthL
        marginWidthB=marginWidthB
        
        line_width=line_width
    elif size.lower()=='half':
        height=480
        width=960  
        fontScale=0.90
        
        fontSmall=fontSmall/0.66*fontScale
        fontMedium=fontMedium/0.66*fontScale
        fontLarge=fontLarge/0.66*fontScale
        
        
        marginWidthR=60
        marginWidthT=marginWidthT/0.66
        marginWidthL=marginWidthL/0.66
        marginWidthB=marginWidthB/0.66
        
        gridThickness=gridThickness/0.66
        
        line_width=line_width/0.66
        dot_size=dot_size/0.66
        
        spaceLegend=spaceLegend/0.66
    else:
        raise ValueError('wrong size. valid are full, half')
    
    fontLegend=fontSmall
    if numTracesWithName(fig)>3:
        fontLegend=fontSmall*0.8
    elif numTracesWithName(fig)>6:
        fontLegend=fontSmall*0.6
        # ...
    elif numTracesWithName(fig)>9:
        opts['legendShow']=False
    
            
    ######Universial configuration
    
    #title
    fig.update_layout(title=None)        
    
    #legend    
    bordercolor='grey'
    if opts['legendPos']=='rightOutside':        
        xAnchor="left"
        yAnchor="top"  
        x=1+spaceLegend
        y=1-spaceLegend/height*width*0.95
        bordercolor=None
        
        
    elif opts['legendPos']=='botRight':        
        xAnchor="right" 
        yAnchor="bottom"  
        x=1-spaceLegend
        y=spaceLegend/height*width*0.95
        
    elif opts['legendPos']=='right':      
        xAnchor="right"      
        yAnchor="middle"  
        x=1-spaceLegend
        y=0.5
    elif opts['legendPos']=='topRight':
        xAnchor="right" 
        yAnchor="top"  
        x=1-spaceLegend
        y=1-spaceLegend/height*width*1.2
        
    elif opts['legendPos']=='topLeft':        
        xAnchor="left" 
        yAnchor="top"  
        x=0+spaceLegend
        y=1-spaceLegend/height*width*0.95
        
        
    elif opts['legendPos']=='left':        
        xAnchor="left" 
        yAnchor="middle"  
        x=0+spaceLegend
        y=0.5
        
    elif opts['legendPos']=='botLeft':        
        xAnchor="left" 
        yAnchor="bottom"  
        x=0+spaceLegend
        y=spaceLegend/height*width*0.95
        
    elif opts['legendPos']=='custom': 
        # xAnchor="left" 
        # yAnchor="top"  
        # x=0.27
        # y=0.8   
        xAnchor="right" 
        yAnchor="bottom"
        x=0.92-spaceLegend
        y=0+spaceLegend
        
        
    fig.update_layout(showlegend=opts['legendShow'])
    fig.update_layout(legend_borderwidth=1)
    fig.update_layout(legend_bgcolor='rgba(255,255,255,0.5)') #lightgrey
    fig.update_layout(legend_bordercolor=bordercolor)
    fig.update_layout(legend_xanchor=xAnchor)
    fig.update_layout(legend_yanchor=yAnchor)
    fig.update_layout(legend_x=x)
    fig.update_layout(legend_y=y)
    fig.update_layout(legend_font_size=fontLegend)
    fig.update_layout(legend_font_color=fontCol)
    
    #axis
    if opts['xTick']!=None:
        fig.update_xaxes(dtick=opts['xTick'])
    if opts['tickvalsX']!=None:
        fig.update_xaxes(tickvals=opts['tickvalsX'])
    if opts['ticktextX']!=None:
        fig.update_xaxes(ticktext=opts['ticktextX'])
    fig.update_xaxes(range=opts['xRange'])
    fig.update_xaxes(showgrid=opts['gridShow'])
    fig.update_xaxes(mirror='all')
    fig.update_xaxes(gridcolor=gridCol)
    fig.update_xaxes(linecolor='black')
    fig.update_xaxes(linewidth=gridThickness)
    fig.update_xaxes(showline=True)
    fig.update_xaxes(ticks="inside")       
    fig.update_xaxes(ticklen=8)    
    fig.update_xaxes(tickfont_size=fontMedium)
    fig.update_xaxes(color=fontCol)
    fig.update_xaxes(title_font_size=fontSmall)
    fig.update_xaxes(zeroline=False)
    fig.update_xaxes(tickformat=opts['xFormat'])
    fig.update_xaxes(ticksuffix=opts['xSuffix'])

    if opts['yTick']!=None:
        fig.update_yaxes(dtick=opts['yTick'])
    if opts['tickvalsY']!=None:
        fig.update_yaxes(tickvals=opts['tickvalsY'])
    if opts['ticktextY']!=None:
        fig.update_yaxes(ticktext=opts['ticktextY'])
    # if opts['yRange']!=None:
    fig.update_yaxes(range=opts['yRange'])
    fig.update_yaxes(showgrid=opts['gridShow'])
    fig.update_yaxes(mirror='all')
    fig.update_yaxes(gridcolor=gridCol)
    fig.update_yaxes(linecolor='black')
    fig.update_yaxes(linewidth=gridThickness)
    fig.update_yaxes(showline=True)
    fig.update_yaxes(ticks="inside")       
    fig.update_yaxes(ticklen=8)    
    fig.update_yaxes(tickfont_size=fontMedium)
    fig.update_yaxes(color=fontCol)
    fig.update_yaxes(title_font_size=fontSmall)
    fig.update_yaxes(zeroline=False)
    fig.update_yaxes(tickformat=opts['yFormat'])
    fig.update_yaxes(ticksuffix=opts['ySuffix'])

    #margin
    fig.update_layout(margin_autoexpand=False)        
    fig.update_layout(margin_l=marginWidthL)
    fig.update_layout(margin_r=marginWidthR)
    fig.update_layout(margin_t=marginWidthT)
    fig.update_layout(margin_b=marginWidthB)
    
    #Draw area
    fig.update_layout(plot_bgcolor='white')
    
    #lines
    fig.update_traces(line_width=line_width, selector=dict(type='scatter'))
    fig.update_traces(marker_size=dot_size, selector=dict(type='scatter'))
    ###Config specific for plot Types
    if plotType==None:     
        ...        
        
    else:
        raise ValueError('specified plotType='+str(plotType)+'. Valid plotTypes'+
                         'is only None')
        
    savePlot(fig,fileName,path=path)
    pio.write_image(fig,filePathName+'.png',format='png',width=width,height=height,scale=2)
    pio.write_image(fig,filePathName+'.pdf',format='pdf',width=width,height=height,scale=2)

    
    
    
    
    
    
    
    
    