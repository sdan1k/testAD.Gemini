"use client";

import { useState } from "react";

const FAQ_CONTENT = {
  title: "Как пользоваться поиском",
  items: [
    {
      question: "Как работает поиск?",
      answer: "Введите текст рекламы или опишите ситуацию — система найдет похожие решения ФАС по смыслу, а не по ключевым словам. Используется семантический поиск на основе ИИ."
    },
    {
      question: "Как использовать фильтры?",
      answer: "• Год — выберите период решений\n• Регион — федеральные округа и субъекты РФ\n• Отрасль — сфера деятельности нарушителя\n• Статья — статья закона о рекламе"
    },
    {
      question: "Что показывает результат?",
      answer: "Каждое дело содержит: описание нарушения, принятое решение ФАС, ссылку на полный документ, информацию о нарушителе."
    },
    {
      question: "Советы по поиску",
      answer: "• Опишите ситуацию своими словами\n• Можно использовать цитаты из рекламы\n• Чем подробнее запрос — тем точнее результат"
    },
    {
      question: "Есть идеи или вопросы. К кому обратиться?",
      answer: "Пишите в Telegram, всегда будем рады помочь:",
      links: [
        { label: "@slm_ct", url: "https://t.me/slm_ct" },
        { label: "@SDan1k", url: "https://t.me/SDan1k" }
      ]
    }
  ]
};

export function HelpButton() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-full shadow-lg hover:bg-primary/90 transition-all hover:scale-105 font-medium"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <path d="M12 17h.01" />
        </svg>
        <span className="hidden sm:inline">Как искать</span>
      </button>

      {/* Modal overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50"
          onClick={() => setIsOpen(false)}
        >
          <div 
            className="bg-card rounded-xl shadow-2xl max-w-lg w-full max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-xl font-semibold">{FAQ_CONTENT.title}</h2>
              <button
                onClick={() => setIsOpen(false)}
                className="p-2 hover:bg-muted rounded-lg transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 6 6 18" />
                  <path d="m6 6 12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4">
              {FAQ_CONTENT.items.map((item, index) => (
                <div key={index} className="space-y-2">
                  <h3 className="font-medium text-primary">{item.question}</h3>
                  <p className="text-muted-foreground text-sm whitespace-pre-line">
                    {item.answer}
                  </p>
                  {item.links && (
                    <div className="flex gap-3 mt-2">
                      {item.links.map((link, linkIndex) => (
                        <a
                          key={linkIndex}
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-primary hover:underline font-medium text-sm"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.161c-.18 1.797-.959 6.158-1.354 8.167-.168.85-.499 1.133-.819 1.16-.696.064-1.225-.46-1.901-.902-1.056-.692-1.653-1.123-2.679-1.799-1.185-.78-.417-1.21.258-1.911.177-.184 3.247-2.977 3.307-3.23.007-.032.015-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.139-5.062 3.345-.479.329-.913.489-1.302.481-.428-.009-1.252-.242-1.865-.442-.751-.244-1.349-.374-1.297-.789.027-.216.324-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.015 3.333-1.386 4.025-1.627 4.477-1.635.099-.002.321.023.465.141.121.1.154.234.169.337.015.103.034.336.019.518z"/>
                          </svg>
                          {link.label}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Footer */}
            <div className="p-4 border-t">
              <button
                onClick={() => setIsOpen(false)}
                className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
              >
                Понятно
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
