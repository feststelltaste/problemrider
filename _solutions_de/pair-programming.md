---
title: Pair Programming
description: Zwei Entwickler arbeiten gemeinsam an einer Aufgabe an einem
  Arbeitsplatz.
category:
- Team
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/pair-programming/
problems:
- knowledge-silos
- tacit-knowledge
- implicit-knowledge
- difficult-developer-onboarding
- lower-code-quality
- reviewer-inexperience
- inadequate-mentoring-structure
- slow-knowledge-transfer
- inappropriate-skillset
- knowledge-dependency
- inexperienced-developers
- skill-development-gaps
- limited-team-learning
- inconsistent-knowledge-acquisition
- avoidance-behaviors
- clever-code
- extended-review-cycles
- individual-recognition-culture
- knowledge-sharing-breakdown
- new-hire-frustration
- poor-teamwork
- procedural-background
- procrastination-on-complex-tasks
- reduced-review-participation
- reduced-team-flexibility
- review-bottlenecks
- reviewer-anxiety
- team-churn-impact
- team-members-not-engaged-in-review-process
- uneven-workload-distribution
- code-review-inefficiency
- incomplete-knowledge
- insufficient-code-review
- staff-availability-issues
- superficial-code-reviews
- inadequate-initial-reviews
- language-barriers
- perfectionist-review-culture
- review-process-avoidance
- review-process-breakdown
- rushed-approvals
- implementation-partner-dependency
layout: solution
lang: de
en_slug: pair-and-mob-programming
related_solutions:
- slug: collaborative-problem-solving
  similarity: 0.85
- slug: knowledge-rotation
  similarity: 0.8
- slug: code-reading-sessions
  similarity: 0.75
- slug: internal-technical-coaching
  similarity: 0.75
- slug: knowledge-sharing-practices
  similarity: 0.75
- slug: structured-onboarding-program
  similarity: 0.75
---

## Description

Pair Programming setzt zwei Entwickler an einem Arbeitsplatz an dieselbe Aufgabe, und Mob Programming erweitert dies auf das gesamte Team, das gemeinsam an einem Problem nach dem anderen arbeitet, und ersetzt Einzelarbeit durch kontinuierlichen, Echtzeit-Wissensaustausch. Dies zählt am meisten in Legacy-Systemen, in denen kritisches Verständnis auf eine oder zwei Personen konzentriert ist, da das Pairing genau auf diesen Modulen mit hohem Bus-Faktor — mit dem kundigen Entwickler als Navigator, während jemand Neues fährt — implizites Verständnis weit schneller überträgt, als es jedes Dokument könnte, und undokumentierte Annahmen in dem Moment offenlegt, in dem der Navigator sie laut aussprechen muss. Die eigentlichen Kosten sind, dass es für jeden, der Durchsatz pro Entwickler misst, wie reduzierte individuelle Ausgabe aussieht, was genau die falsche Linse für Arbeit ist, deren tatsächlicher Ertrag eine Codebasis ist, die nicht mehr vom Gedächtnis einer Person abhängt.

## How to Apply ◆

> In Legacy-Systemen, in denen kritisches Wissen auf eine kleine Anzahl von Personen konzentriert ist, sind Pair und Mob Programming die direktesten Interventionen, um dieses Wissen zu übertragen, bevor es verloren geht.

- Priorisieren Sie Pairing auf den Teilen der Legacy-Codebasis mit dem höchsten Bus-Faktor — den Modulen, die wirklich nur eine Person versteht — und behandeln Sie diese Sitzungen ausdrücklich als Wissenstransfer, nicht nur als Entwicklung.
- Verwenden Sie Strong-Style-Pairing (Navigator diktiert die Absicht, Fahrer tippt), wenn ein erfahrener Entwickler mit Legacy-Wissen zusammen mit einem Junior arbeitet, der neu im System ist; der Senior muss jede Annahme artikulieren, die er sonst stillschweigend anwenden würde.
- Wenden Sie Mob Programming auf besonders gefährliche oder komplexe Legacy-Module an, bei denen mehrere Perspektiven benötigt werden und ein Fehler breite Auswirkungen haben könnte — das gesamte Team, das gemeinsam eine fragile gemeinsame Komponente durcharbeitet, ist oft sicherer, als wenn eine Einzelperson sie allein anfasst.
- Beim Debuggen obskurer Legacy-Fehler paarweise statt allein untersuchen; der Navigator erkennt oft das relevante Muster (eine bekannte Eigenheit des Systems, eine undokumentierte Abhängigkeit) weit schneller als ein einzelner Entwickler, der einen unbekannten Stack durchverfolgt.
- Rotieren Sie Paare bewusst über unbekannte Module hinweg, statt standardmäßig die Person einzusetzen, die einen Bereich „schon kennt" — das Ziel ist, einzelne Wissenspunkte zu beseitigen, nicht kurzfristige Effizienz zu optimieren.
- Beschränken Sie Pairing-Sitzungen auf zwei bis drei Stunden am Stück; Legacy-Code-Erkundung ist kognitiv intensiv, und ausgedehntes Pairing ohne Pausen produziert abnehmende Erträge.
- Nutzen Sie Mob-Programming-Sitzungen zum Onboarding neuer Entwickler in die Legacy-Codebasis; ein Gruppendurchgang durch ein Schlüsselmodul im Mob-Stil, mit dem Neuling als Fahrer, legt implizites Wissen schneller offen, als es jede schriftliche Dokumentation könnte.
- Verfolgen Sie mit einer einfachen Matrix, welche Teammitglieder an welchen Modulen gepaart haben; wenn sich immer dieselben Paare bilden, verbessert sich die Wissensverteilung nicht, und Rotation muss durchgesetzt werden.

## Tradeoffs ⇄

> Pair und Mob Programming erfordern die Investition der Zeit von zwei oder mehr Entwicklern in eine Aufgabe, was besonders umstritten ist in Legacy-Teams, die bereits unter Druck stehen — aber die Kosten eines Wissenssilos, der dauerhaft wird, sind typischerweise weit höher.

**Vorteile:**

- Verteilt Wissen über notorisch isolierte Legacy-Module auf mehr Teammitglieder und reduziert direkt das Risiko, dass kritisches Wissen mit einem langjährigen Entwickler zur Tür hinausgeht.
- Fängt legacy-spezifische Fehler — falsche Annahmen über externes Systemverhalten, missverstandenen gemeinsamen Zustand, implizite Reihenfolgebeschränkungen — im Moment der Änderung statt in Produktion ab.
- Produziert bessere Designs für Legacy-Modifikationen, weil der Navigator den breiteren Kontext bewahrt, während sich der Fahrer auf die Implementierung konzentriert, was den Tunnelblick reduziert, den einzelne Entwickler entwickeln, wenn sie allein in unbekanntem Code arbeiten.
- Beschleunigt das Onboarding in das Legacy-System dramatisch im Vergleich zur Einzelerkundung, weil der Pairing-Partner Echtzeit-Erklärungen der Eigenheiten und Geschichte des Systems liefert.
- Reduziert Länge und Komplexität nachfolgender Code-Reviews, da von einem Paar geschriebener Code bereits kontinuierlich während des Schreibens überprüft wurde.

**Kosten und Risiken:**

- In Teams mit nur ein oder zwei Personen, die ein kritisches Legacy-Modul verstehen, bedeutet das Pairing dieses Experten für Wissenstransfer, dass er kurzfristig individuell weniger produziert, was durchsatzfokussierte Manager zurückweisen könnten.
- Persönlichkeitskonflikte, Erfahrungsunterschiede und unterschiedliche Arbeitsstile werden in Pairing-Situationen beim Umgang mit frustrierendem Legacy-Code verstärkt; der Stress der Arbeit in schwierigem Code kann in die Zusammenarbeit überschwappen.
- Pairing an Legacy-Code ohne Tests oder Dokumentation ist mental erschöpfender als Pairing an gut strukturiertem Greenfield-Code; Sitzungen müssen kürzer und Pausen häufiger sein.
- In Organisationen, die individuelle Entwicklerproduktivität messen, ist Pair Programming auf individueller Ebene unsichtbar — zwei Entwickler, die ein Ticket schließen, wirken wie halbe Produktivität, selbst wenn die Qualitätsergebnisse besser sind.
- Remote-Pairing führt zusätzliche Reibung ein beim Umgang mit Legacy-Code, der lokale Umgebungseinrichtung, proprietäres Tooling oder Zugang zu institutionellen Systemen erfordert, die sich schwer über Screen-Sharing-Tools teilen lassen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Pair und Mob Programming Wissenskonzentration und Qualitätsprobleme in Legacy-System-Kontexten adressieren.

Die versicherungsmathematische Berechnungs-Engine eines Versicherungsunternehmens war elf Jahre lang von einem einzigen Entwickler betreut worden. Als dieser Entwickler seinen Ruhestand ankündigte, organisierte das Team eine Reihe von Mob-Programming-Sitzungen, in denen der ausscheidende Entwickler navigierte, während jüngere Teammitglieder die Implementierung geplanter Features fuhren. Über vier Monate erwarben drei Entwickler genug Verständnis der Berechnungslogik, um unabhängig daran zu arbeiten. Die Mob-Sitzungen produzierten auch die vollständigste Dokumentation, die das Modul je gehabt hatte, direkt im Code und in den schriftlichen Aufzeichnungen der Sitzungen selbst festgehalten.

Eine Regierungsbehörde, die ein Legacy-Steuerverarbeitungssystem betrieb, musste einen neuen digitalen Identitätsanbieter integrieren. Die Integration betraf ein Modul, das zwei Entwickler Jahre zuvor mittels eines maßgeschneiderten Protokolls gebaut hatten, das niemand sonst verstand. Statt die Integration allein den ursprünglichen Autoren zuzuweisen, führte das Team eine Woche Mob-Programming-Sitzungen mit allen vier verfügbaren Entwicklern durch, einschließlich der beiden ohne vorherige Erfahrung mit dem Modul. Der Mob-Ansatz zwang die ursprünglichen Autoren, ihre Protokollentscheidungen laut zu erklären, legte drei undokumentierte Grenzfälle offen, die in Produktion zu Fehlern geführt hätten, und hinterließ dem Team vier Personen, die die Integration künftig warten konnten.

Ein Fintech-Startup hatte seinen Zahlungsverarbeitungsdienst über Jahre expedienter Ergänzungen wachsen lassen, bis er sowohl kritisch als auch fragil war. Als das Team eine kontrollierte Modernisierungsanstrengung begann, führte es eine Richtlinie ein, an jeder Änderung des Zahlungsdienstes zu paaren, unabhängig davon, wie klein sie war. Die Navigatorrolle wurde unter den erfahrensten Entwicklern des Teams rotiert. Über sechs Monate produzierte diese Richtlinie zwei Ergebnisse: Die Anhäufung neuer Schulden im Zahlungsdienst verlangsamte sich merklich, weil der Navigator konsequent schnelle Fixes infrage stellte, und das Wissen um die Eigenheiten des Dienstes verbreitete sich von den zwei ursprünglichen Autoren auf fünf zusätzliche Entwickler, die nun auf Produktionsvorfälle reagieren konnten, ohne eskalieren zu müssen.
