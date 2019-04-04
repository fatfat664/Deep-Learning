import matplotlib.pyplot as plt
import matplotlib.patches as patch

def plot1():
    SGD_leg = patch.Patch(color='blue', label='SGD')
    SGDmom_leg = patch.Patch(color='red', label='SGD with momentum')
    AdaGrad_leg = patch.Patch(color='orange', label='AdaGrad')
    Adam_leg = patch.Patch(color='green', label='Adam')

    x = []
    for i in range(0, 51):
        x.append(i)

    SGD = [3652.83,
    3649.68,
    3645.22,
    3636.38,
    3603.68,
    3537.67,
    3448.42,
    3355.54,
    3268.94,
    3178.76,
    3090.52,
    3033.14,
    2999.73,
    2974.64,
    2955.72,
    2940.24,
    2922.17,
    2899.08,
    2880.26,
    2865.03,
    2851.85,
    2841.65,
    2832.16,
    2819.81,
    2800.46,
    2780.07,
    2763.94,
    2744.32,
    2731.17,
    2720.82,
    2711.88,
    2704.90,
    2699.84,
    2693.54,
    2689.79,
    2686.09,
    2681.89,
    2679.52,
    2675.58,
    2673.09,
    2668.96,
    2667.34,
    2663.98,
    2662.10,
    2658.80,
    2658.24,
    2656.56,
    2653.98,
    2651.63,
    2650.09,
    2650.41]

    SGDmom = [3481.02,
    3016.35,
    2862.90,
    2748.58,
    2674.48,
    2651.70,
    2632.08,
    2580.65,
    2559.33,
    2543.30,
    2537.11,
    2531.41,
    2523.85,
    2519.16,
    2518.00,
    2515.75,
    2511.87,
    2508.83,
    2507.22,
    2503.97,
    2504.13,
    2501.97,
    2502.52,
    2501.43,
    2499.90,
    2497.59,
    2497.87,
    2495.98,
    2495.76,
    2495.78,
    2494.51,
    2494.19,
    2494.08,
    2493.84,
    2495.03,
    2493.18,
    2494.36,
    2490.92,
    2492.08,
    2493.36,
    2492.71,
    2490.87,
    2491.49,
    2491.63,
    2491.18,
    2490.77,
    2489.37,
    2490.68,
    2491.07,
    2491.44,
    2489.23]

    AdaGrad = [3648.23,
    3640.65,
    3632.72,
    3623.54,
    3612.84,
    3600.27,
    3586.77,
    3571.60,
    3554.75,
    3537.87,
    3521.33,
    3504.09,
    3489.76,
    3474.81,
    3461.27,
    3447.65,
    3433.27,
    3422.72,
    3411.90,
    3401.89,
    3390.25,
    3380.20,
    3370.10,
    3360.46,
    3352.86,
    3344.64,
    3335.97,
    3328.94,
    3319.86,
    3311.74,
    3305.10,
    3298.20,
    3291.37,
    3285.43,
    3277.38,
    3269.67,
    3264.65,
    3257.50,
    3249.14,
    3244.61,
    3238.54,
    3231.89,
    3226.82,
    3221.42,
    3216.49,
    3211.52,
    3207.27,
    3203.28,
    3197.88,
    3193.90,
    3188.47]

    Adam = [3636.24,
    3523.90,
    3364.16,
    3226.62,
    3136.25,
    3074.48,
    3029.39,
    2988.72,
    2954.31,
    2922.89,
    2900.52,
    2879.51,
    2857.55,
    2838.86,
    2823.90,
    2811.66,
    2798.65,
    2787.59,
    2779.26,
    2767.85,
    2758.99,
    2753.81,
    2744.53,
    2740.01,
    2733.39,
    2724.32,
    2717.03,
    2708.19,
    2697.89,
    2691.65,
    2686.17,
    2680.15,
    2674.10,
    2667.56,
    2663.95,
    2658.73,
    2656.70,
    2651.12,
    2644.93,
    2641.93,
    2637.69,
    2635.09,
    2633.55,
    2629.44,
    2627.67,
    2624.23,
    2620.23,
    2617.52,
    2617.42,
    2613.48,
    2613.53]

    plt.plot(x, SGD, color='blue')
    plt.plot(x, SGDmom, color='red')
    plt.plot(x, AdaGrad, color='orange')
    plt.plot(x, Adam, color='green')
    plt.legend(handles=[SGD_leg, SGDmom_leg, AdaGrad_leg, Adam_leg])

    plt.xlabel('Epochs')
    plt.ylabel('Training Cost')
    plt.title('Epochs vs Training Cost')

    plt.show()

def plot2():
    SGD_leg = patch.Patch(color='blue', label='SGD')
    SGDmom_leg = patch.Patch(color='red', label='SGD with momentum')
    AdaGrad_leg = patch.Patch(color='orange', label='AdaGrad')
    Adam_leg = patch.Patch(color='green', label='Adam')

    x = []
    for i in range(0, 51, 10):
        x.append(i)

    SGD = [0.16, 0.59, 0.70, 0.79, 0.81, 0.83]
    SGDmom = [0.52, 0.90, 0.92, 0.93, 0.93, 0.93]
    AdaGrad = [0.24, 0.41, 0.46, 0.52, 0.55, 0.57]
    Adam = [0.28, 0.70, 0.76, 0.82, 0.85, 0.86]

    plt.plot(x, SGD, color='blue')
    plt.plot(x, SGDmom, color='red')
    plt.plot(x, AdaGrad, color='orange')
    plt.plot(x, Adam, color='green')
    plt.legend(handles=[SGD_leg, SGDmom_leg, AdaGrad_leg, Adam_leg])

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Epochs vs Validation Accuracy')

    plt.show()

def plot3(x,y,z):
    plt.plot(x, y)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Steps vs Cumulative reward')

    plt.show()

#plot1()
#plot2()
plot3()