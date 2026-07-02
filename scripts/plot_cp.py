import pandas as pd
import matplotlib.pyplot as plt

def plot_cp_distribution(csv_filename):
    print(f"Lendo os dados do arquivo: {csv_filename}...")
    
    # Carregar os dados do CSV
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_filename}' não foi encontrado.")
        return

    # Separar os dados em Extradorso (y >= 0) e Intradorso (y < 0)
    # A ordenação por 'x' é importante para que a linha do gráfico não fique "ziguezagueando"
    extradorso = df[df['y'] >= 0].sort_values(by='x')
    intradorso = df[df['y'] < 0].sort_values(by='x')

    # Configurar a figura
    plt.figure(figsize=(10, 6))

    # Plotar as curvas
    plt.plot(extradorso['x'], extradorso['Cp'], 
             label='Extradorso (Superior)', 
             color='blue', 
             marker='o', 
             markersize=4, 
             linestyle='-', 
             linewidth=1.5)
             
    plt.plot(intradorso['x'], intradorso['Cp'], 
             label='Intradorso (Inferior)', 
             color='red', 
             marker='s', 
             markersize=4, 
             linestyle='-', 
             linewidth=1.5)

    # Inverter o eixo Y (Padrão ouro em plotagens de aerodinâmica)
    plt.gca().invert_yaxis()

    # Configurações de design do gráfico
    plt.title('Distribuição do Coeficiente de Pressão ($C_p$)', fontsize=14, pad=15)
    plt.xlabel('Posição na corda ($x/c$)', fontsize=12)
    plt.ylabel('Coeficiente de Pressão ($C_p$)', fontsize=12)
    
    # Adicionar grade e legenda
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='best', fontsize=11)
    
    # Ajustar as margens
    plt.tight_layout()

    # Salvar a imagem e mostrar na tela
    output_img = csv_filename.replace('.csv', '.png')
    plt.savefig(output_img, dpi=300)
    print(f"Gráfico salvo como: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    # Substitua pelo nome do arquivo gerado pelo seu código Rust
    # Exemplo: 'malha_naca0012_Cp.csv'
    NOME_DO_ARQUIVO = "job_files/biconvex_fine/Cp_distribution.csv" 
    
    plot_cp_distribution(NOME_DO_ARQUIVO)