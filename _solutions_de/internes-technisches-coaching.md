---
title: Internes technisches Coaching
description: Jemandem explizite, geschützte Zeit geben, um die technische Praxis
  des Teams zu heben, indem er mit Menschen zusammenarbeitet statt sie nur zu unterrichten.
category:
- Team
- Process
- Culture
problems:
- inexperienced-developers
- skill-development-gaps
- limited-team-learning
- slow-knowledge-transfer
- misunderstanding-of-oop
- procedural-programming-in-oop-languages
- cargo-culting
- inappropriate-skillset
- inadequate-mentoring-structure
- reviewer-inexperience
- clever-code
- inconsistent-execution
- defensive-coding-practices
- incomplete-knowledge
- inconsistent-knowledge-acquisition
- knowledge-dependency
- author-frustration
- difficult-to-understand-code
- extended-research-time
- high-turnover
- insufficient-design-skills
- legacy-skill-shortage
- mentor-burnout
- new-hire-frustration
- reduced-team-flexibility
- reviewer-anxiety
- implementation-partner-dependency
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: internal-technical-coaching
related_solutions:
- slug: pair-and-mob-programming
  similarity: 0.75
- slug: technical-skills-development
  similarity: 0.7
- slug: code-reading-sessions
  similarity: 0.7
- slug: communities-of-practice
  similarity: 0.7
- slug: structured-onboarding-program
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
---

## Description

Internes technisches Coaching ist die bewusste Zuweisung einer Person — mit geschützter Zeit und explizitem Mandat —, die technische Praxis eines Teams zu heben, indem sie mit dessen Mitgliedern an deren echter Arbeit zusammenarbeitet. Es unterscheidet sich von Training darin, dass es in der Codebasis statt im Klassenzimmer geschieht, und von Mentoring darin, dass es die Praxis des Teams statt die Karriere einer Einzelperson anvisiert. Die Unterscheidung, die es funktionieren lässt, ist geschützte Zeit: Ein erfahrener Entwickler, von dem auch erwartet wird, Features zu liefern, wird immer Features liefern, weil das gemessen wird. In Legacy-Kontexten ist der Bedarf akut, und die üblichen Abhilfen passen nicht. Externe Kurse lehren Muster, die Greenfield-Bedingungen annehmen, und die Fähigkeiten, die tatsächlich zählen — unbekannten Code lesen, Abhängigkeiten sicher brechen, entscheiden, wann man nicht refaktorieren sollte — werden gelernt, indem man sie neben jemandem tut, der sie schon kann.

## How to Apply ◆

> Die Techniken, die Legacy-Arbeit handhabbar machen, werden selten irgendwo gelehrt und fast nie aufgeschrieben; sie verbreiten sich durch die Zusammenarbeit mit jemandem, der sie hat.

- **Schützen Sie die Zeit explizit** — ein erklärter Anteil der Kapazität des Coaches, typischerweise zwanzig bis fünfzig Prozent, aus Lieferverpflichtungen herausgenommen. Coaching, von dem erwartet wird, neben einer vollen Lieferlast zu geschehen, geschieht nicht, und das Versagen ist unsichtbar, weil alle beschäftigt sind.
- **Coachen Sie an der echten Arbeit des Teams**, nicht an Übungen. Der Wert liegt darin zu zeigen, wie man dieses Legacy-Modul, diese untestbare Klasse, diese unklare Anforderung angeht — der Transfer überlebt die Abstraktion in ein Spielzeugbeispiel nicht.
- **Arbeiten Sie mit Menschen, nicht an ihnen.** Der Coach paart bei echten Aufgaben, übernimmt gelegentlich die Tastatur und gibt sie zurück, und lässt die andere Person produktiv kämpfen. Ein Coach, der das Problem löst, hat eine Lösung produziert statt einer Fähigkeit.
- **Wählen Sie eine kleine Anzahl von Praktiken** und verfolgen Sie sie, bis sie haften bleiben. Ein Coach, der gleichzeitig Testen, Refactoring, Domänenmodellierung und Review-Qualität fördert, erreicht oberflächliche Vertrautheit mit allen vieren. Eine ordentlich übernommene Praxis ist vier versuchte wert.
- **Wählen Sie den Coach nach Lehrfähigkeit, nicht Seniorität.** Der beste Entwickler in einem Team ist häufig ein schlechter Coach, weil seine Expertise implizit geworden ist und er sie nicht zerlegen kann. Die Bereitschaft, im Tempo einer anderen Person zu arbeiten, zählt mehr als rohe Fähigkeit.
- Kombinieren Sie Formate bewusst: **Pairing für Tiefe, Code-Reading-Sitzungen für Breite und kurze fokussierte Workshops** für eine spezifische Technik. Unterschiedliches Wissen überträgt sich durch unterschiedliche Kanäle, und Pairing allein erreicht zu wenige Menschen.
- Geben Sie dem Coach ein **Mandat, Praxis zu ändern, nicht nur zu beraten**. Ein Coach, der nur empfehlen kann, wird in dem Moment ignoriert, in dem Termindruck aufkommt. Das Mandat sollte explizit und dem Team bekannt sein.
- **Vereinbaren Sie beobachtbare Ziele** von Anfang an — mehr Menschen, die in einem gegebenen Subsystem arbeiten können, Tests, die Änderungen in einem gegebenen Bereich begleiten, Review-Kommentare, die von Stil zu Substanz wechseln. Coaching ohne Ziele wird zu einer unbestimmten Rolle, deren Wert niemand einschätzen kann und die in der ersten Budgetprüfung gestrichen wird.
- **Rotieren Sie, wer gecoacht wird**, statt sich auf die neuesten Mitglieder zu konzentrieren. Langjährige Entwickler in einem Legacy-System tragen oft die am stärksten verfestigten Gewohnheiten, und sie sind auch diejenigen, deren Praxis alle anderen am meisten formt.
- **Planen Sie, dass der Coach überflüssig wird.** Das Erfolgsmaß ist, dass die Praxis ohne ihn fortbesteht, was bedeutet, bewusst zu übergeben — gecoachte Entwickler die nächste Gruppe coachen zu lassen.

## Tradeoffs ⇄

> Coaching hebt die Praxis eines ganzen Teams dauerhaft, auf Kosten eines erheblichen Anteils der Kapazität einer erfahrenen Person und mit Ergebnissen, die langsam und schwer zuzuordnen sind.

**Vorteile:**

- Fähigkeiten übertragen sich in dem Kontext, in dem sie genutzt werden, was weit dauerhafter ist als Klassenzimmertraining, das vom Lernenden auf die echte Codebasis übertragen werden muss.
- Die spezifischen Kompetenzen, die Legacy-Arbeit erfordert — sicheres Abhängigkeitsbrechen, inkrementelles Refactoring, Lesen unbekannten Codes — werden gelehrt, und diese sind von externem Training im Wesentlichen nicht verfügbar.
- Die Praxis wird über das Team hinweg konsistenter, was die Divergenz verringert, wie dieselben Probleme in verschiedenen Ecken des Systems gelöst werden.
- Die Review-Qualität verbessert sich, während mehr Menschen zu substanziellem Review fähig werden, was die kleine Gruppe entlastet, die es derzeit trägt.
- Nachgeahmte Praktiken werden untersucht, weil zur Aufgabe eines Coaches gehört zu fragen, warum etwas so gemacht wird — eine Frage, die im Team niemand mehr stellt.

**Kosten und Risiken:**

- Ein erheblicher Anteil der Kapazität eines erfahrenen Entwicklers verlässt die Lieferung, was sofort spürbar ist, während die Vorteile über Quartale eintreffen.
- Die Rolle ist schwer zu bewerten, und Coaching ist daher in jeder Budget- oder Personalprüfung gefährdet, trotz effektiv zu sein.
- Ein schlecht gewählter Coach kann seine eigenen Präferenzen als Teamstandards verfestigen, was schlimmer ist als gar kein Coaching, falls diese Präferenzen veraltet oder dogmatisch sind.
- Coaching, das einem Team aufgedrückt wird, das nicht danach gefragt hat, erzeugt höflichen Widerstand, und der Coach verbringt seine Zeit mit Menschen, die sich fügen, statt zu lernen.
- Der Coach kann selbst zu einer Abhängigkeit werden, falls der Übergabeschritt übersprungen wird, was die Praxis des Teams wieder von einer Person abhängig macht.

## How It Could Be

Ein elfköpfiges Team, das eine Energiehandelsplattform pflegte, hatte zwei Entwickler, die Tests für die Legacy-Preismodule schreiben konnten, und neun, die es nicht konnten, mit dem Ergebnis, dass die meisten Änderungen ungetestet ausgeliefert wurden. Formales Training war versucht worden: ein zweitägiger Kurs zu Unit-Testing, nach dem sich nichts änderte, weil die präsentierten Techniken injizierbare Abhängigkeiten annahmen und der tatsächliche Code keine hatte. Das Team wies stattdessen einem der beiden fähigen Entwickler 40 Prozent seiner Zeit als Coach für sechs Monate zu, mit einem Ziel: Jede Änderung am Preis-Subsystem wird mit einem Test ausgeliefert. Er paarte mit jedem Entwickler an dessen eigenen Tickets und lehrte Extract-and-Override und Characterization Testing an den tatsächlichen Klassen, die sie änderten. Nach sechs Monaten schrieben neun von elf ungefragt Tests für dieses Subsystem, und die Testzahl im Bereich war von 12 auf über 400 gestiegen.

Dieselbe Coaching-Vereinbarung brachte etwas ans Licht, wonach das Team nicht gesucht hatte. Pairing über das gesamte Team hinweg offenbarte, dass vier Entwickler unabhängig voneinander eine Datumsverarbeitungsroutine neu implementierten, weil jeder glaubte, das gemeinsame Utility sei defekt. Es war nicht defekt; es war undokumentiert, und seine Parameterreihenfolge war überraschend. Der teamübergreifende Blick des Coaches war es, der das Muster sichtbar machte, da jeder Entwickler es allein angetroffen und allein umgangen hatte. Die Parameter zu benennen und sechs Zeilen Dokumentation zu schreiben beseitigte eine Quelle duplizierter Logik, die sich seit zwei Jahren still ausgebreitet hatte.
