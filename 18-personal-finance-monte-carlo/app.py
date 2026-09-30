import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Future Paths Lab",page_icon="🎲",layout="wide")
st.title("Future Paths Lab — Monte Carlo financial outcomes")
st.warning("This educational simulation is conditional on your assumptions. It does not predict the future or provide financial advice.")

with st.sidebar:
    starting=st.number_input("Starting balance",0.0,100_000_000.0,25000.0,1000.0)
    contribution=st.number_input("Monthly contribution",0.0,1_000_000.0,750.0,50.0)
    years=st.slider("Years",1,60,25);annual_return=st.slider("Expected annual arithmetic return",-.10,.25,.07,.005)
    annual_vol=st.slider("Annual volatility",0.0,.60,.16,.01);target=st.number_input("Target balance",0.0,1_000_000_000.0,750000.0,25000.0)
    inflation=st.slider("Annual inflation assumption",0.0,.15,.025,.005);simulations=st.select_slider("Simulations",[1000,2500,5000,10000,20000],value=10000)
    distribution=st.selectbox("Monthly return distribution",["Normal","Heavy-tailed Student t"]);seed=st.number_input("Random seed",0,999999,42)


@st.cache_data
def run(starting,contribution,years,annual_return,annual_vol,simulations,distribution,seed):
    rng=np.random.default_rng(seed);months=years*12;mu=annual_return/12;sigma=annual_vol/np.sqrt(12)
    if distribution=="Normal":returns=rng.normal(mu,sigma,(months,simulations))
    else:
        raw=rng.standard_t(5,(months,simulations));returns=mu+sigma*raw/np.sqrt(5/3)
    balances=np.empty((months+1,simulations));balances[0]=starting
    for month in range(months):balances[month+1]=np.maximum(0,balances[month]*(1+returns[month])+contribution)
    return balances


paths=run(starting,contribution,years,annual_return,annual_vol,simulations,distribution,seed);ending=paths[-1];q=np.quantile(ending,[.1,.25,.5,.75,.9]);real=ending/(1+inflation)**years
deterministic=starting*(1+annual_return/12)**(years*12)+contribution*((1+annual_return/12)**(years*12)-1)/(annual_return/12) if annual_return!=0 else starting+contribution*years*12
tabs=st.tabs(["Outcome distribution","Sample paths","Sensitivity","Sequence risk","Math & limitations"])
with tabs[0]:
    cols=st.columns(4);cols[0].metric("Median ending balance",f"${q[2]:,.0f}");cols[1].metric("10th–90th percentile",f"${q[0]:,.0f}–${q[4]:,.0f}");cols[2].metric("Target reached",f"{(ending>=target).mean():.1%}");cols[3].metric("Median in today's dollars",f"${np.median(real):,.0f}")
    st.plotly_chart(px.histogram(x=ending,nbins=70,labels={"x":"Ending balance"},title="Distribution of simulated outcomes"),width="stretch")
    st.dataframe(pd.DataFrame({"Percentile":[10,25,50,75,90],"Ending balance":q}).style.format({"Ending balance":"${:,.0f}"}),width="stretch",hide_index=True)
with tabs[1]:
    rng=np.random.default_rng(seed+1);ids=rng.choice(simulations,min(80,simulations),replace=False);timeline=np.arange(years*12+1)/12;sample=pd.DataFrame(paths[:,ids],index=timeline);sample.index.name="Year"
    fig=px.line(sample,title="Sample simulated paths");fig.update_traces(line_width=.7,opacity=.25,showlegend=False);fig.add_scatter(x=timeline,y=np.median(paths,axis=1),name="Median",line_width=4);st.plotly_chart(fig,width="stretch")
    st.metric("Deterministic compound-growth result",f"${deterministic:,.0f}");st.caption("The deterministic line hides dispersion and sequence effects even when it uses the same expected return.")
with tabs[2]:
    vols=np.linspace(0,.40,9);rows=[]
    for v in vols:
        e=run(starting,contribution,years,annual_return,float(v),3000,distribution,seed+7)[-1];rows.append({"Volatility":v,"10th percentile":np.quantile(e,.1),"Median":np.median(e),"90th percentile":np.quantile(e,.9),"Target probability":np.mean(e>=target)})
    sensitivity=pd.DataFrame(rows);st.plotly_chart(px.line(sensitivity,x="Volatility",y=["10th percentile","Median","90th percentile"],markers=True,title="Outcome range versus volatility"),width="stretch");st.plotly_chart(px.line(sensitivity,x="Volatility",y="Target probability",markers=True),width="stretch")
with tabs[3]:
    returns=paths[1:]/np.maximum(paths[:-1],1)-1-contribution/np.maximum(paths[:-1],1);example=returns[:,0];ordered_high=np.sort(example)[::-1];ordered_low=np.sort(example)
    def replay(r):
        b=starting
        for x in r:b=max(0,b*(1+x)+contribution)
        return b
    st.dataframe(pd.DataFrame({"Same return set":["Original order","Best returns first","Worst returns first"],"Ending balance":[replay(example),replay(ordered_high),replay(ordered_low)]}).style.format({"Ending balance":"${:,.0f}"}),width="stretch",hide_index=True)
    st.caption("With ongoing contributions, ordering changes how much capital experiences later returns. Withdrawals can make sequence risk even more consequential.")
with tabs[4]:
    st.markdown(r"""Each month samples a random return with mean approximately $\mu_{annual}/12$ and volatility $\sigma_{annual}/\sqrt{12}$, then applies $B_{t+1}=\max(0,B_t(1+r_t)+C)$. Repeating this process approximates a conditional outcome distribution.

Normal and Student-t returns are simplified models: real returns can be autocorrelated, regime-dependent, skewed, fat-tailed, and affected by taxes, fees, inflation, and behavior. Expected returns and volatility are unknowable and nonstationary. Percentiles and target frequencies are simulation outputs—not confidence statements about reality.""")

