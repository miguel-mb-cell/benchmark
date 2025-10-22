const translate = require('@iamtraction/google-translate');

const textToTranslate = process.argv[2];
const targetLang = process.argv[3] || 'en';
const sourceLang = process.argv[4] || 'auto'; // <-- NOVO ARGUMENTO

if (!textToTranslate) {
  process.exit(1); // Sai se não houver texto
}

// Objeto de opções dinâmico
const options = {
  to: targetLang,
  from: sourceLang // <-- USA O NOVO ARGUMENTO (padrão 'auto' se não for passado)
};

translate(textToTranslate, options)
  .then(res => {
    console.log(res.text); // Imprime o resultado para o Python capturar
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });