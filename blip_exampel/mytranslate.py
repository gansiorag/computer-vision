# git clone https://github.com/argosopentech/argos-translate.git
# cd argos-translate
# python3 -m pip install .

from argostranslate import package, translate

print('Загружаются модели переводчика...')

package.install_from_path('ru_en.argosmodel')
package.install_from_path('en_ru.argosmodel')

installed_languages = translate.load_installed_languages()
translation_ru_en = installed_languages[1].get_translation(
    installed_languages[0])
translation_en_ru = installed_languages[0].get_translation(
    installed_languages[1])

print('Модели переводчика успешно загружены')


def ruen(text):
    return translation_ru_en.translate(text)


def enru(text):
    return translation_en_ru.translate(text)
