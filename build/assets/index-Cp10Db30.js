import{c as i,k as r,r as l,j as t}from"./index-OxGwEg3m.js";import{R as m,P as p}from"./RenderInPortalIfExists-D-C7LcEp.js";const c="/assets/balloon-0.Czj7AKwE.png",g="/assets/balloon-1.CNvFFrND.png",d="/assets/balloon-2.DTvC6B1t.png",x="/assets/balloon-3.CgSk4tbL.png",f="/assets/balloon-4.mbtFrzxf.png",b="/assets/balloon-5.CSwkUfRA.png",n=300,h=121,s=20,u=80,C=1e3,N=r`
  from {
    transform: translateY(calc(100vh + ${n}px));
  }

  to {
    transform: translateY(0);
  }
`,w=i("img",{target:"earwcwy0"})(({theme:a})=>({position:"fixed",top:"-300px",marginLeft:`${-121/2}px`,zIndex:a.zIndices.balloons,left:`${Math.random()*(u-s)+s}vw`,animationDelay:`${Math.random()*C}ms`,height:`${n}px`,width:`${h}px`,pointerEvents:"none",animationDuration:"750ms",animationName:N,animationTimingFunction:"ease-in",animationDirection:"normal",animationIterationCount:1,opacity:1}),""),o=30,e=[c,g,d,x,f,b],I=e.length,$=({particleType:a})=>t.jsx(w,{src:e[a]}),j=({scriptRunId:a})=>t.jsx(m,{children:t.jsx(p,{className:"stBalloons","data-testid":"stBalloons",scriptRunId:a,numParticleTypes:I,numParticles:o,ParticleComponent:$})}),v=l.memo(j);export{o as NUM_BALLOONS,v as default};
