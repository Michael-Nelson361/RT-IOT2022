# For working with the data
import pandas as pd
import numpy as np

# For visualizing the data
import matplotlib.pyplot as plt
import seaborn as sns

def q1(df):
    """
    <summary>
    
    Parameters:
    -----------
    
    Returns:
    --------
    
    """
    # print question part 1
    print("Across duration and size of packets, where do the attack patterns tend to lie?")
    sns.relplot(
        data=df,
        x='flow_duration',
        y='flow_pkts_payload.avg',
        hue='Attack_type',
        kind='scatter',
        col='traffic_type'
    ).set_axis_labels("Session Duration (in milliseconds)", "Average Payload Size of Packets")
    plt.show()
    print("We see here that the attack patterns tend to be very similar to normal traffic, and that there doesn't seem to be much relation between the average size and the duration. However, we can see that the attack durations do tend to be shorter, which makes sense since a threat actor may want to keep his or her patterns less noticeable.\n")
    
    # print question part 2
    print("Where do the types of services tend to lie?")
    sns.relplot(
        data=df,
        x='flow_duration',
        y='flow_pkts_payload.avg',
        hue='service',
        kind='scatter',
        col='traffic_type'
    ).set_axis_labels("Session Duration (in milliseconds)", "Average Payload Size of Packets")
    plt.show()
    
    # print question part 3
    print("Where do the protocols tend to lie?")
    sns.relplot(
        data=df,
        x='flow_duration',
        y='flow_pkts_payload.avg',
        hue='proto',
        kind='scatter',
        col='traffic_type'
    ).set_axis_labels("Session Duration (in milliseconds)", "Average Payload Size of Packets")
    plt.show()
    print("We see that as above, threat actors attempt to keep much of their network traffic as similar as possible to regular network traffic in order to go unnoticed.")
    
    return None

def q2(df):
    """
    <summary>
    
    Parameters:
    -----------
    
    Returns:
    --------
    
    """
    print("Which services are attack patterns usually targeting?")
    sns.relplot(
        data=df,
        x='flow_duration',
        y='flow_pkts_payload.avg',
        hue='traffic_type',
        palette={"Attack": "red", "Normal": "green"},
        kind='scatter',
        col='service',
        col_wrap=5,
        height=3,
        aspect=1  # Adjust aspect ratio to spread plots wider
    ).set_axis_labels("Session Duration", "Average Payload Size of Packets")
    plt.show()
    
    # Step 1: Calculate counts
    grouped = df.groupby(['service', 'traffic_type']).size().reset_index(name='counts')

    # Step 2: Calculate total counts for each service to use for percentage calculation
    total_counts = grouped.groupby('service')['counts'].transform('sum')

    # Convert counts to percentages
    grouped['percentage'] = (grouped['counts'] / total_counts) * 100

    # Step 3: Create the plot
    sns.catplot(
        kind='bar',  # Use bar type to show percentages
        data=grouped,
        x='service',
        y='percentage',
        hue='traffic_type',
        palette={"Attack": "red", "Normal": "green"}
    )

    # Adding more descriptive labels
    plt.xlabel('Service')
    plt.ylabel('Percentage')
    plt.yticks(list(range(0,110,10)))
    plt.title('Percentage of Traffic Types by Service')
    plt.grid(alpha=0.4,axis='y')

    plt.show()
    
    print("Overall, of the services targeted, they seem to be largely the 'none' type, RADIUS, SSH, and DHCP. Being that RADIUS and SSH are especially used in authentication, this makes sense as a threat actor would prefer to have continued access to the network.")
    
    
    return None

def q3(df):
    """
    <summary>
    
    Parameters:
    -----------
    
    Returns:
    --------
    
    """
    print("What is the distribution of attacks to normal traffic?")
    sns.catplot(
        data=df,
        kind='count',
        x='traffic_type',
        palette={
            'Attack':'red',
            'Normal':'green'
        }
    )
    plt.show()
    
    sns.catplot(
        data=df,
        kind='count',
        x='Attack_type',
        hue='traffic_type',
        palette={
            'Attack':'red',
            'Normal':'green'
        }
    )
    plt.xticks(rotation=-45,ha="left")
    plt.show()
    
    print("How does this compare with the services and protocols being used?")
    # Step 1: Calculate counts
    counts = df.groupby(['Attack_type', 'service', 'traffic_type']).size().reset_index(name='counts')

    # Step 2: Compute percentages
    total_counts = counts.groupby(['Attack_type', 'traffic_type'])['counts'].transform('sum')
    counts['percentage'] = (counts['counts'] / total_counts) * 100

    # Step 3: Create the plot using calculated percentages
    sns.catplot(
        data=counts,
        kind='bar',  # Changed to bar to show percentages more clearly
        x='Attack_type',
        y='percentage',
        hue='service',
        col='traffic_type',
        # Uncomment and set the palette if needed
        # palette={'Attack':'red', 'Normal':'green'}
    ).set_xticklabels(rotation=-45, ha="left")
    plt.yticks(list(range(0,110,10)))
    plt.show()
    print("We can see that the primarily targeted service is most of all generic and not at particular services, and the most used is the DOS_SYN_Hping.")
    
    
    
    return None
