const translate = require('@iamtraction/google-translate');

// Pega o texto do primeiro argumento da linha de comando
const textToTranslate = process.argv[2];
const targetLang = process.argv[3] || 'en'; // Padrão 'en' se não for especificado

if (!textToTranslate) {
  process.exit(1); // Sai se não houver texto
}

translate(textToTranslate, { to: targetLang })
  .then(res => {
    console.log(res.text); // Imprime o resultado para o Python capturar
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });