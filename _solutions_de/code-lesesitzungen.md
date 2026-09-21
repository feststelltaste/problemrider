---
title: Code-Lesesitzungen
description: Gemeinsames, lautes Lesen bestehenden Codes als geplante Gruppenaktivität
  — der schnellste Weg, Verständnis für ein System zu verbreiten, das niemand vollständig
  versteht.
category:
- Team
- Communication
- Code
problems:
- difficult-to-understand-code
- clever-code
- knowledge-silos
- slow-knowledge-transfer
- limited-team-learning
- inexperienced-developers
- misunderstanding-of-oop
- procedural-programming-in-oop-languages
- knowledge-gaps
- incomplete-knowledge
- difficult-developer-onboarding
- legacy-system-documentation-archaeology
- cargo-culting
- skill-development-gaps
- bloated-class
- copy-paste-programming
- global-state-and-side-effects
- inconsistent-knowledge-acquisition
- knowledge-dependency
- maintenance-bottlenecks
- superficial-code-reviews
- code-duplication
- complex-implementation-paths
- defensive-coding-practices
- extended-research-time
- hidden-side-effects
- inappropriate-skillset
- information-decay
- insufficient-design-skills
- legacy-skill-shortage
- mentor-burnout
- monolithic-functions-and-classes
- new-hire-frustration
- over-reliance-on-utility-classes
- poor-encapsulation
- procrastination-on-complex-tasks
- reduced-team-flexibility
- reviewer-anxiety
- team-churn-impact
- team-members-not-engaged-in-review-process
- implementation-partner-dependency
layout: solution
lang: de
en_slug: code-reading-sessions
related_solutions:
- slug: pair-and-mob-programming
  similarity: 0.75
- slug: code-review-process-reform
  similarity: 0.7
- slug: structured-onboarding-program
  similarity: 0.7
- slug: collaborative-problem-solving
  similarity: 0.7
- slug: internal-technical-coaching
  similarity: 0.7
- slug: code-hotspot-analysis
  similarity: 0.7
---

## Description

Eine Code-Lesesitzung ist ein geplantes Meeting, in dem eine Gruppe ein Stück bestehenden Codes gemeinsam, laut, liest und herausfindet, was er tut. Sie kehrt das übliche Code-Review um: Nichts wird vorgeschlagen, nichts wird beurteilt, und der Code ist typischerweise Jahre alt. Der Zweck ist Verständnis, und der Grund, warum es besser funktioniert als individuelles Lesen, ist, dass das Verstehen von Legacy-Code ein Prozess der Bildung und Verwerfung von Hypothesen ist, der weit schneller geschieht, wenn mehrere Personen es laut tun. Teams nutzen dies unter, weil das Lesen von Code keine sichtbare Ausgabe hat und sich unproduktiv anfühlt im Vergleich zum Schreiben. Aber in einem Legacy-System ist der begrenzende Faktor für fast jede Aufgabe Verständnis statt Tippen, und das Verständnis ist aktuell so verteilt, dass jede Person ein anderes Fragment hält. Eine Lesesitzung ist der günstigste verfügbare Mechanismus, um diese Fragmente zusammenzuführen.

## How to Apply ◆

> Der Code, der es am meisten wert ist, gemeinsam gelesen zu werden, ist der Code, den alle meiden — was genau der Code ist, den niemand jemals allein lesen wird.

- **Wählen Sie das Ziel bewusst**: ein Modul, das mehrere Personen bald berühren müssen, ein Bereich mit konzentriertem Wissen oder Code, der weiterhin Defekte produziert. Hotspot-Daten — häufig geändert, häufig in Bugs verwickelt — wählen gute Kandidaten, wenn die Intuition unsicher ist.
- Halten Sie Sitzungen **kurz und begrenzt**: sechzig bis neunzig Minuten, und ein klar abgegrenztes Stück Code. Der Versuch, ein ganzes Subsystem abzudecken, produziert eine Führung statt eines Verständnisses.
- **Lesen Sie den Code, fassen Sie ihn nicht zusammen.** Projizieren Sie ihn, gehen Sie ihn Zeile für Zeile durch, und lassen Sie Menschen alles fragen. In dem Moment, in dem es zu einer vorbereiteten Präsentation von demjenigen wird, der ihn bereits kennt, hört der Mechanismus auf zu funktionieren — der Wert liegt in den Fragen der Gruppe, nicht in der Erzählung des Experten.
- **Ermutigen Sie die naive Frage explizit.** „Warum ist das hier?" und „was passiert, wenn das null ist?" von jemandem, der mit dem Code nicht vertraut ist, decken regelmäßig echte Defekte und Annahmen auf, die die vertrauten Leser aufgehört hatten zu sehen.
- Lassen Sie jemanden **Notizen machen und committen**, als Kommentare, als kurzes Dokument oder als Diagramm. Eine Sitzung, die nichts Dauerhaftes produziert, muss für jede neue Person wiederholt werden. Die Notizen sind außerdem das Nächste, was das Modul jemals an Dokumentation haben wird.
- **Erfassen Sie die Fragen, die niemand beantworten konnte.** Dies sind die wertvollsten Punkte der Sitzung: Sie markieren die Teile des Systems, in denen Wissen tatsächlich verloren gegangen ist, und sie bilden die Agenda für Untersuchung oder die nächste Sitzung.
- **Rotieren Sie, wer den Code wählt**, sodass Sitzungen abdecken, was jede Person undurchdringlich findet, statt was eine Person interessant findet.
- Nutzen Sie es bewusst für das **Onboarding**: Ein neuer Mitarbeiter, der vier oder fünf Lesesitzungen über die Hauptsubsysteme besucht, erwirbt eine funktionierende Karte weit schneller als durch das Lesen von Dokumentation oder die Anweisung, die Codebasis zu erkunden.
- Halten Sie es **getrennt von Review und von Kritik.** Der gelesene Code ist üblicherweise schlecht — das ist oft, warum er gewählt wurde. Eine Sitzung, die zu einer Kritik abwesender Autoren wird, wird unsicher für denjenigen, der den nächste Woche zu lesenden Code geschrieben hat.

## Tradeoffs ⇄

> Gemeinsames Lesen verbreitet Verständnis schnell und findet echte Defekte, auf Kosten der Zeit mehrerer Personen für eine Aktivität ohne unmittelbares Liefergut.

**Vorteile:**

- Verständnis verbreitet sich schnell über das Team, was direkt die Konzentration von Wissen adressiert, die Legacy-Wartung fragil macht.
- Defekte und unsichere Annahmen tauchen während des Lesens auf, besonders durch Fragen von Menschen, die mit dem Code nicht vertraut sind, zu weit geringeren Kosten als sie in Produktion zu finden.
- Dokumentation wird als Nebenprodukt produziert, geschrieben von Menschen, die gerade entdeckt haben, was unklar war, und genau auf dieses Publikum ausgerichtet.
- Onboarding beschleunigt sich erheblich, weil ein neuer Entwickler Kontext über das echte System gewinnt statt über seine dokumentierte Abstraktion.
- Weniger erfahrene Entwickler beobachten, wie erfahrene Hypothesen über unvertrauten Code bilden, was eine Fähigkeit ist, die sonst fast nie explizit gelehrt wird.

**Kosten und Risiken:**

- Mehrere Personen verbringen eine Stunde oder mehr ohne Liefergut, was schwierig gegenüber jemandem zu rechtfertigen ist, der Output misst, und das Erste ist, was unter Druck fallengelassen wird.
- Sitzungen können in Kritik vergangener Entwickler abdriften, was unproduktiv ist und Menschen defensiv über Code macht, den sie selbst geschrieben haben.
- Ohne Notizen verdunstet das Verständnis, und die Sitzung muss für die nächste Person wiederholt werden, was die Praxis schnell verschwenderisch erscheinen lässt.
- Ein dominanter Experte kann die Sitzung in einen Vortrag verwandeln, der Struktur vermittelt, aber nicht die Begründung, die das Wissen nutzbar macht.
- Verständnis zerfällt, wenn niemand danach im Code arbeitet, sodass Sitzungen am besten kurz vor der tatsächlichen Berührung des Bereichs geplant werden.

## How It Could Be

Ein sechsköpfiges Team, das eine Abonnement-Abrechnungsplattform pflegte, hatte einen Entwickler, der die Proration-Logik verstand — ungefähr 1.200 Zeilen, die sich über elf Jahre angehäuft hatten — und jede Änderung, die sie berührte, wartete in seiner Warteschlange. Sie planten vier neunzigminütige Lesesitzungen über zwei Wochen. In der zweiten Sitzung fragte ein Entwickler, der vier Monate zuvor beigetreten war, warum ein bestimmter Zweig einen Tag subtrahierte, und niemand konnte antworten. Untersuchung fand, dass er einen Zeitzonenbehandlungsfehler anderswo im System kompensierte, und dass die Kompensation für zwei der elf unterstützten Länder falsch war — ein Defekt, der geschätzt drei Jahre lang still falsche Rechnungen produziert hatte. Am Ende der vier Sitzungen konnten drei Entwickler im Modul arbeiten, und die während der Sitzungen gemachten Notizen wurden zur ersten Dokumentation, die es je gehabt hatte.

Dasselbe Team übernahm Lesesitzungen für Onboarding. Ihr vorheriger Ansatz war ein schriftliches Architekturdokument und die Anweisung, die Codebasis zu erkunden, wonach neue Entwickler ungefähr vier Monate brauchten, um eine unbeaufsichtigte Änderung an einem Kernmodul vorzunehmen. Neue Mitarbeiter besuchen nun eine Lesesitzung pro Woche für ihre ersten sechs Wochen, die die fünf Hauptsubsysteme abdeckt. Die Zeit bis zur ersten unbeaufsichtigten Änderung in einem Kernmodul sank auf ungefähr sechs Wochen. Ein unerwarteter Effekt war, dass die langjährigen Entwickler die Sitzungen ebenfalls wertvoll fanden: In der Sitzung über die Zahlungsgateway-Integration entdeckten zwei Personen, die jeweils Jahre daran gearbeitet hatten, dass sie inkompatible Überzeugungen darüber hatten, wie Wiederholungen mit Idempotenzschlüsseln interagierten, und einer von ihnen lag falsch.
