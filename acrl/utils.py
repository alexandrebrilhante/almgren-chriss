import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.compat.python import zip_longest
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_2cols

import acrl.synthetic as sca


def generate_table(left_col, right_col, table_title):
    col_headers = None

    if right_col:
        if len(right_col) < len(left_col):
            right_col += [(" ", " ")] * (len(left_col) - len(right_col))
        elif len(right_col) > len(left_col):
            left_col += [(" ", " ")] * (len(right_col) - len(left_col))

        right_col = [("%-21s" % ("  " + k), v) for k, v in right_col]

        gen_stubs_right, gen_data_right = zip_longest(*right_col)
        gen_table_right = SimpleTable(
            gen_data_right,
            col_headers,
            gen_stubs_right,
            title=table_title,
            txt_fmt=fmt_2cols,
        )

    else:
        gen_table_right = []

    gen_stubs_left, gen_data_left = zip_longest(*left_col)
    gen_table_left = SimpleTable(
        gen_data_left, col_headers, gen_stubs_left, title=table_title, txt_fmt=fmt_2cols
    )
    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table


def get_env_param():
    env = sca.MarketEnvironment()

    fp_title = "Financial Parameters"
    fp_left_col = [
        ("Annual Volatility:", ["{:.0f}%".format(env.anv * 100)]),
        ("Daily Volatility:", ["{:.1f}%".format(env.dpv * 100)]),
    ]

    fp_right_col = [
        ("Bid-Ask Spread:", ["{:.3f}".format(env.basp)]),
        ("Daily Trading Volume:", ["{:,.0f}".format(env.dtv)]),
    ]

    acp_title = "Almgren and Chriss Model Parameters"
    acp_left_col = [
        ("Total Number of Shares to Sell:", ["{:,}".format(env.total_shares)]),
        ("Starting Price per Share:", ["${:.2f}".format(env.startingPrice)]),
        ("Price Impact for Each 1% of Daily Volume Traded:", ["${}".format(env.eta)]),
        ("Number of Days to Sell All the Shares:", ["{}".format(env.liquidation_time)]),
        ("Number of Trades:", ["{}".format(env.num_n)]),
    ]

    acp_right_col = [
        ("Fixed Cost of Selling per Share:", ["${:.3f}".format(env.epsilon)]),
        ("Trader's Risk Aversion:", ["{}".format(env.llambda)]),
        ("Permanent Impact Constant:", ["{}".format(env.gamma)]),
        ("Single Step Variance:", ["{:.3f}".format(env.singleStepVariance)]),
        ("Time Interval between trades:", ["{}".format(env.tau)]),
    ]

    fp_table = generate_table(fp_left_col, fp_right_col, fp_title)
    acp_table = generate_table(acp_left_col, acp_right_col, acp_title)

    return fp_table, acp_table


def plot_price_model(seed=0, num_days=1000):
    env = sca.MarketEnvironment()
    env.reset(seed)

    price_hist = np.zeros(num_days)

    for i in range(num_days):
        _, _, _, info = env.step(i)
        price_hist[i] = info.price

    print("Average Stock Price: ${:,.2f}".format(price_hist.mean()))
    print("Standard Deviation in Stock Price: ${:,.2f}".format(price_hist.std()))

    price_df = pd.DataFrame(data=price_hist, columns=["Stock"], dtype="float64")

    ax = price_df.plot(colormap="cool", grid=False)
    ax.set_facecolor(color="k")
    ax = plt.gca()

    yNumFmt = mticker.StrMethodFormatter("${x:,.2f}")

    ax.yaxis.set_major_formatter(yNumFmt)

    plt.ylabel("Stock Price")
    plt.xlabel("days")
    plt.show()


def get_optimal_vals(lq_time=60, nm_trades=60, tr_risk=1e-6, title=""):
    env = sca.MarketEnvironment()
    env.reset(liquid_time=lq_time, num_trades=nm_trades, lamb=tr_risk)

    if title == "":
        title = "AC Optimal Strategy"
    else:
        title = "AC Optimal Strategy for " + title

    E = env.get_AC_expected_shortfall(env.total_shares)
    V = env.get_AC_variance(env.total_shares)
    U = env.compute_AC_utility(env.total_shares)

    left_col = [
        ("Number of Days to Sell All the Shares:", ["{}".format(env.liquidation_time)]),
        ("Half-Life of The Trade:", ["{:,.1f}".format(1 / env.kappa)]),
        ("Utility:", ["${:,.2f}".format(U)]),
    ]

    right_col = [
        (
            "Initial Portfolio Value:",
            ["${:,.2f}".format(env.total_shares * env.startingPrice)],
        ),
        ("Expected Shortfall:", ["${:,.2f}".format(E)]),
        ("Standard Deviation of Shortfall:", ["${:,.2f}".format(np.sqrt(V))]),
    ]

    value_table = generate_table(left_col, right_col, title)

    return value_table


def get_min_param():
    min_impact = get_optimal_vals(
        lq_time=250, nm_trades=250, tr_risk=1e-17, title="Minimum Impact"
    )

    min_var = get_optimal_vals(
        lq_time=1, nm_trades=1, tr_risk=0.0058, title="Minimum Variance"
    )

    return min_impact, min_var


def get_crfs(trisk):
    tr_st = "{:.0e}".format(trisk)

    lnum = tr_st.split("e")[0]

    lexp = tr_st.split("e")[1]

    if np.abs(np.int(lexp)) < 10:
        lexp = lexp.replace("0", "", 1)

    an_st = "$\lambda = " + lnum + " \\times 10^{" + lexp + "}$"

    if trisk >= 1e-7 and trisk <= 4e-7:
        xcrf = 0.94
        ycrf = 2.5
        scrf = 0.1
    elif trisk > 4e-7 and trisk <= 9e-7:
        xcrf = 0.9
        ycrf = 2.5
        scrf = 0.06
    elif trisk > 9e-7 and trisk <= 1e-6:
        xcrf = 0.85
        ycrf = 2.5
        scrf = 0.06
    elif trisk > 1e-6 and trisk < 2e-6:
        xcrf = 1.2
        ycrf = 2.5
        scrf = 0.06
    elif trisk >= 2e-6 and trisk < 3e-6:
        xcrf = 0.8
        ycrf = 2.5
        scrf = 0.06
    elif trisk >= 3e-6 and trisk < 4e-6:
        xcrf = 0.7
        ycrf = 2.5
        scrf = 0.08
    elif trisk >= 4e-6 and trisk < 7e-6:
        xcrf = 1.4
        ycrf = 2.0
        scrf = 0.08
    elif trisk >= 7e-6 and trisk <= 1e-5:
        xcrf = 4.5
        ycrf = 1.5
        scrf = 0.08
    elif trisk > 1e-5 and trisk <= 2e-5:
        xcrf = 7.0
        ycrf = 1.1
        scrf = 0.08
    elif trisk > 2e-5 and trisk <= 5e-5:
        xcrf = 12.0
        ycrf = 1.1
        scrf = 0.08
    elif trisk > 5e-5 and trisk <= 1e-4:
        xcrf = 30
        ycrf = 0.99
        scrf = 0.08
    else:
        xcrf = 1
        ycrf = 1
        scrf = 0.08

    return an_st, xcrf, ycrf, scrf


def plot_efficient_frontier(tr_risk=1e-6):
    env = sca.MarketEnvironment()
    env.reset(lamb=tr_risk)

    tr_E = env.get_AC_expected_shortfall(env.total_shares)
    tr_V = env.get_AC_variance(env.total_shares)

    E = np.array([])
    V = np.array([])
    U = np.array([])

    num_points = 7000

    lambdas = np.linspace(1e-7, 1e-4, num_points)

    for llambda in lambdas:
        env.reset(lamb=llambda)
        E = np.append(E, env.get_AC_expected_shortfall(env.total_shares))
        V = np.append(V, env.get_AC_variance(env.total_shares))
        U = np.append(U, env.compute_AC_utility(env.total_shares))

    cm = plt.cm.get_cmap("gist_rainbow")

    sc = plt.scatter(V, E, s=20, c=U, cmap=cm)

    plt.colorbar(sc, label="AC Utility", format=mticker.StrMethodFormatter("${x:,.0f}"))

    ax = plt.gca()
    ax.set_facecolor("k")

    ymin = E.min() * 0.7
    ymax = E.max() * 1.1

    plt.ylim(ymin, ymax)

    yNumFmt = mticker.StrMethodFormatter("${x:,.0f}")
    xNumFmt = mticker.StrMethodFormatter("{x:,.0f}")

    ax.yaxis.set_major_formatter(yNumFmt)
    ax.xaxis.set_major_formatter(xNumFmt)

    plt.xlabel("Variance of Shortfall")
    plt.ylabel("Expected Shortfall")

    an_st, xcrf, ycrf, scrf = get_crfs(tr_risk)

    plt.annotate(
        an_st,
        xy=(tr_V, tr_E),
        xytext=(tr_V * xcrf, tr_E * ycrf),
        color="w",
        size="large",
        arrowprops=dict(facecolor="cyan", shrink=scrf, width=3, headwidth=10),
    )

    plt.show()


def round_trade_list(trl):
    trl_rd = np.around(trl)

    res = np.around(trl.sum() - trl_rd.sum())

    if res != 0:
        idx = trl_rd.nonzero()[0][-1]
        trl_rd[idx] += res

    return trl_rd


def plot_trade_list(lq_time=60, nm_trades=60, tr_risk=1e-6, show_trl=False):
    env = sca.MarketEnvironment()
    env.reset(liquid_time=lq_time, num_trades=nm_trades, lamb=tr_risk)

    trade_list = env.get_trade_list()

    new_trl = np.insert(trade_list, 0, 0)

    df = pd.DataFrame(
        data=list(range(nm_trades + 1)), columns=["Trade Number"], dtype="float64"
    )

    df["Stocks Sold"] = new_trl

    df["Stocks Remaining"] = (np.ones(nm_trades + 1) * env.total_shares) - np.cumsum(
        new_trl
    )

    _, axes = plt.subplots(nrows=1, ncols=2)

    df.iloc[1:].plot.scatter(
        x="Trade Number",
        y="Stocks Sold",
        c="Stocks Sold",
        colormap="gist_rainbow",
        alpha=1,
        sharex=False,
        s=50,
        colorbar=False,
        ax=axes[0],
    )

    axes[0].plot(
        df["Trade Number"].iloc[1:],
        df["Stocks Sold"].iloc[1:],
        linewidth=2.0,
        alpha=0.5,
    )

    axes[0].set_facecolor(color="k")

    yNumFmt = mticker.StrMethodFormatter("{x:,.0f}")

    axes[0].yaxis.set_major_formatter(yNumFmt)
    axes[0].set_title("Trading List")

    df.plot.scatter(
        x="Trade Number",
        y="Stocks Remaining",
        c="Stocks Remaining",
        colormap="gist_rainbow",
        alpha=1,
        sharex=False,
        s=50,
        colorbar=False,
        ax=axes[1],
    )

    axes[1].plot(df["Trade Number"], df["Stocks Remaining"], linewidth=2.0, alpha=0.5)
    axes[1].set_facecolor(color="k")

    yNumFmt = mticker.StrMethodFormatter("{x:,.0f}")

    axes[1].yaxis.set_major_formatter(yNumFmt)
    axes[1].set_title("Trading Trajectory")

    plt.subplots_adjust(wspace=0.4)
    plt.show()

    print("\nNumber of Shares Sold: {:,.0f}\n".format(new_trl.sum()))

    if show_trl:
        rd_trl = round_trade_list(new_trl)

        df2 = pd.DataFrame(
            data=list(range(nm_trades + 1)), columns=["Trade Number"], dtype="float64"
        )

        df2["Stocks Sold"] = rd_trl
        df2["Stocks Remaining"] = (
            np.ones(nm_trades + 1) * env.total_shares
        ) - np.cumsum(rd_trl)

        return df2.style.hide_index().format(
            {
                "Trade Number": "{:.0f}",
                "Stocks Sold": "{:,.0f}",
                "Stocks Remaining": "{:,.0f}",
            }
        )


def implement_trade_list(seed=0, lq_time=60, nm_trades=60, tr_risk=1e-6):
    env = sca.MarketEnvironment()
    env.reset(seed=seed, liquid_time=lq_time, num_trades=nm_trades, lamb=tr_risk)

    trl = env.get_trade_list()
    trade_list = round_trade_list(trl)

    env.start_transactions()

    price_hist = np.array([])

    for trade in trade_list:
        action = trade / env.shares_remaining

        _, _, _, info = env.step(action)

        price_hist = np.append(price_hist, info.exec_price)

        if info.done:
            print(
                "Implementation Shortfall: ${:,.2f} \n".format(
                    info.implementation_shortfall
                )
            )
            break

    price_df = pd.DataFrame(data=price_hist, columns=["Stock"], dtype="float64")

    ax = price_df.plot(colormap="cool", grid=False)
    ax.set_facecolor(color="k")
    ax.set_title("Impacted Stock Price")

    ax = plt.gca()

    yNumFmt = mticker.StrMethodFormatter("${x:,.2f}")

    ax.yaxis.set_major_formatter(yNumFmt)

    plt.plot(price_hist, "o")
    plt.ylabel("Stock Price")
    plt.xlabel("Trade Number")
    plt.show()


def get_av_std(lq_time=60, nm_trades=60, tr_risk=1e-6, trs=100):
    env = sca.MarketEnvironment()
    env.reset(liquid_time=lq_time, num_trades=nm_trades, lamb=tr_risk)

    trl = env.get_trade_list()
    trade_list = round_trade_list(trl)
    shortfall_hist = np.array([])

    for episode in range(trs):
        if (episode + 1) % 100 == 0:
            print("Episode [{}/{}]".format(episode + 1, trs), end="\r", flush=True)

        env.reset(seed=episode, liquid_time=lq_time, num_trades=nm_trades, lamb=tr_risk)
        env.start_transactions()

        for trade in trade_list:
            action = trade / env.shares_remaining
            _, _, _, info = env.step(action)

            if info.done:
                shortfall_hist = np.append(
                    shortfall_hist, info.implementation_shortfall
                )
                break

    print("Average Implementation Shortfall: ${:,.2f}".format(shortfall_hist.mean()))
    print(
        "Standard Deviation of the Implementation Shortfall: ${:,.2f}".format(
            shortfall_hist.std()
        )
    )

    plt.plot(shortfall_hist, "cyan", label="")
    plt.xlim(0, trs)

    ax = plt.gca()
    ax.set_facecolor("k")
    ax.set_xlabel("Episode", fontsize=15)
    ax.set_ylabel("Implementation Shortfall (US $)", fontsize=15)

    ax.axhline(shortfall_hist.mean(), 0, 1, color="m", label="Average")

    yNumFmt = mticker.StrMethodFormatter("${x:,.0f}")

    ax.yaxis.set_major_formatter(yNumFmt)

    plt.legend()
    plt.show
