import{c as r,k as m,r as p,j as n}from"./index-OxGwEg3m.js";import{R as l,P as c}from"./RenderInPortalIfExists-D-C7LcEp.js";const d="/assets/flake-0.DgWaVvm5.png",f="/assets/flake-1.B2r5AHMK.png",g="/assets/flake-2.BnWSExPC.png",o=150,x=150,u=10,$=90,h=4e3,a=(t,e=0)=>Math.random()*(t-e)+e,k=()=>m`
  from {
    transform:
      translateY(0)
      rotateX(${a(360)}deg)
      rotateY(${a(360)}deg)
      rotateZ(${a(360)}deg);
  }

  to {
    transform:
      translateY(calc(100vh + ${o}px))
      rotateX(0)
      rotateY(0)
      rotateZ(0);
  }
`,I=r("img",{target:"ekdfb790"})(({theme:t})=>({position:"fixed",top:"-150px",marginLeft:`${-150/2}px`,zIndex:t.zIndices.balloons,left:`${a($,u)}vw`,animationDelay:`${a(h)}ms`,height:`${o}px`,width:`${x}px`,pointerEvents:"none",animationDuration:"3000ms",animationName:k(),animationTimingFunction:"ease-in",animationDirection:"normal",animationIterationCount:1,opacity:1}),""),s=100,i=[d,f,g],P=i.length,S=({particleType:t})=>n.jsx(I,{src:i[t]}),j=function({scriptRunId:t}){return n.jsx(l,{children:n.jsx(c,{className:"stSnow","data-testid":"stSnow",scriptRunId:t,numParticleTypes:P,numParticles:s,ParticleComponent:S})})},v=p.memo(j);export{s as NUM_FLAKES,v as default};
