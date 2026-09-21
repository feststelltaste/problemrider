---
title: Exploratives Testen
description: Eine erfahrene Person das System bewusst und ohne Skript in zeitlich
  begrenzten Sitzungen mit dokumentierten Befunden untersuchen lassen, um zu entdecken,
  was niemand zu spezifizieren bedachte.
category:
- Testing
- Process
problems:
- insufficient-testing
- poor-test-coverage
- high-defect-rate-in-production
- missing-end-to-end-tests
- testing-complexity
- increased-manual-testing-effort
- regression-bugs
- reduced-feature-quality
- hidden-side-effects
- requirements-ambiguity
- inadequate-requirements-gathering
- quality-degradation
- cache-invalidation-problems
- deadlock-conditions
- improper-event-listener-management
- increased-risk-of-bugs
- negative-brand-perception
- partial-bug-fixes
- stack-overflow-errors
- unreleased-resources
- user-trust-erosion
layout: solution
lang: de
en_slug: exploratory-testing
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
- slug: characterization-tests
  similarity: 0.75
- slug: fuzz-testing
  similarity: 0.7
- slug: regression-testing
  similarity: 0.7
- slug: code-reading-sessions
  similarity: 0.7
---

## Description

Exploratives Testen ist die strukturierte Untersuchung eines Systems durch eine Person, die gleichzeitig Tests entwirft und ausführt und aus jedem Ergebnis lernt, was als Nächstes zu versuchen ist. Es ist kein Ad-hoc-Klicken und kein Ersatz für automatisierte Tests. Es besetzt eine Lücke, die Automatisierung konstruktionsbedingt nicht füllen kann: Ein automatisierter Test kann nur etwas prüfen, das sich jemand bereits ausgedacht hat, während die Defekte, die in einem Legacy-System wirklich zählen, überproportional die sind, die niemand antizipiert hat. Ein geschickter Erkunder folgt dem tatsächlichen Verhalten des Systems — einer seltsamen Nachricht, einer langsamen Antwort, einem Zustand, der nicht erreichbar sein sollte — auf Pfaden, die keine Spezifikation beschreibt. Legacy-Systeme belohnen dies ungewöhnlich gut, weil sie Jahrzehnte angesammelten Verhaltens enthalten, das kein aktuelles Dokument beschreibt und keine Testsuite abdeckt, sodass es viel zu entdecken gibt.

## How to Apply ◆

> Die produktivsten explorativen Sitzungen in einem Legacy-System zielen auf die Bereiche, in denen niemand sagen kann, was das korrekte Verhalten ist, weil dort noch nie jemand einen Test schreiben konnte.

- **Arbeiten Sie in zeitlich begrenzten Sitzungen mit einer festgelegten Mission** — sechzig bis neunzig Minuten, mit einer Ein-Satz-Aufgabe wie „untersuchen, was mit einer Bestellung passiert, wenn der Zahlungsanbieter mitten in der Transaktion ein Timeout hat". Eine unbegrenzte Sitzung wird unfokussiert; eine Mission ohne Zeitrahmen wird zu einer Untersuchung ohne Ende.
- **Dokumentieren Sie fortlaufend, was Sie getan und gefunden haben**, einschließlich Pfaden, die nichts ergaben. Die Notizen sind es, was die Sitzung reproduzierbar, berichtbar und anschließend in automatisierte Tests umwandelbar macht.
- **Folgen Sie den Hinweisen des Systems statt einem Plan.** Eine Antwort, die vier Sekunden braucht, während andere fünfzig Millisekunden brauchen, eine Fehlermeldung, die eine Komponente erwähnt, die nicht beteiligt sein sollte, ein Feld, das mehr Zeichen akzeptiert als es sollte — das sind die Spuren, und ihnen zu folgen ist die Fertigkeit.
- **Variieren Sie bewusst entlang bekannter Bruchlinien**: Grenzen, leere und maximale Eingaben, ungewöhnliche Sequenzen, Unterbrechung und Fortsetzung, gleichzeitiger Zugriff und der Zurück-Button. In Legacy-Systemen sind Sequenzen, die die Designer nicht antizipiert haben, durchweg die ergiebigste Quelle.
- **Verwenden Sie produktionsähnliche Daten.** Das Erkunden anhand sauberer synthetischer Datensätze findet weit weniger, weil das interessante Verhalten durch die historischen und fehlerhaften Datensätze ausgelöst wird, die echte Systeme enthalten.
- **Wandeln Sie Ihre Funde in automatisierte Tests um.** Ein durch Exploration gefundener Defekt sollte kein zweites Mal durch Exploration auffindbar sein. Die Exploration findet den Fall; der automatisierte Test hält ihn gefunden.
- **Wählen Sie Missionen nach Risiko**, nicht nach Abdeckung: kürzlich geänderte Bereiche, Code ohne Testabdeckung, Integrationspunkte mit externen Systemen und alles, was zuvor Vorfälle verursacht hat.
- **Lassen Sie andere Personen als den Autor erkunden.** Das mentale Modell des Autors ist es, das das Verhalten produziert hat, sodass er strukturell am wenigsten geeignet ist, dessen blinde Flecken zu finden. Eine Entwicklerin mit jemandem aus Support oder Betrieb zu paaren ist häufig sehr produktiv.
- **Berichten Sie Befunde als Beobachtungen mit Belegen**, nicht als Urteile. In einem Legacy-System ist oft echt unklar, ob ein Verhalten ein Defekt oder eine langjährige beabsichtigte Eigenheit ist, und die Aufgabe der Exploration ist es, dies sichtbar zu machen, damit jemand entscheiden kann.
- **Planen Sie es regelmäßig ein**, nicht nur vor Releases. Exploration, die ausschließlich als Vor-Release-Gate genutzt wird, wird zu einer gehetzten Regressionsprüfung, was das ist, worin sie am schlechtesten ist.

## Tradeoffs ⇄

> Exploration findet die Defekte, die Automatisierung strukturell nicht finden kann, auf Kosten geschickter Menschenzeit und Ergebnissen, die weder wiederholbar noch vorhersagbar sind.

**Vorteile:**

- Es findet Defekte, die keine automatisierte Suite jemals enthalten würde, weil diese Tests von jemandem geschrieben worden sein müssten, der den Fall bereits kannte.
- Es bringt undokumentiertes Verhalten an die Oberfläche, das in einem Legacy-System einen beträchtlichen Teil der tatsächlichen Spezifikation ausmacht.
- Es erfordert keine Testinfrastruktur und funktioniert daher in Systemen, in denen automatisiertes Testen derzeit unpraktisch ist — oft die Systeme, die Testen am dringendsten brauchen.
- Befunde werden direkt in automatisierte Tests umgewandelt, sodass die Praxis als Nebeneffekt eine Suite aufbaut.
- Es bringt Usability- und Kohärenzprobleme an die Oberfläche, die jede funktionale Prüfung bestehen, da ein Mensch Verwirrung bemerkt, wo eine Assertion es nicht tut.

**Kosten und Risiken:**

- Es verbraucht bei jeder Gelegenheit geschickte Menschenzeit und kann nicht bei jeder Änderung automatisch laufen, was es als Regressionsmechanismus ungeeignet macht.
- Ergebnisse hängen stark von der Fähigkeit und dem Wissen des Erkunders ab, sodass die Praxis schwer zu planen oder zu garantieren ist.
- Abdeckung ist konstruktionsbedingt unbekannt. Eine Sitzung, die nichts findet, kann bedeuten, dass der Bereich einwandfrei ist oder dass der Erkunder am falschen Ort gesucht hat, und beides ist nicht zu unterscheiden.
- Befunde können in einem Legacy-System mehrdeutig sein, und zu entscheiden, ob langjähriges seltsames Verhalten ein Defekt ist, verbraucht Zeit von Personen, die es womöglich auch nicht wissen.
- Ohne dokumentierte Notizen und Umwandlung in automatisierte Tests werden dieselben Defekte wiederholt neu entdeckt, und die Praxis erwirbt den Ruf, Aufwand statt Fortschritt zu erzeugen.

## How It Could Be

Ein Team, das ein Krankenhausterminsystem pflegte, hatte 78 Prozent automatisierte Testabdeckung und eine anhaltende Rate an Produktionsdefekten, die die Suite nie fing. Sie führten wöchentliche neunzigminütige explorative Sitzungen mit Missionen ein, die aus kürzlich geänderten und historisch problematischen Bereichen gewählt wurden. Die Mission der dritten Sitzung war „was passiert, wenn ein Termin verschoben wird, während ihn eine Ärztin bearbeitet". Der Erkunder fand, dass das zweite Speichern das erste stillschweigend überschrieb, dass die Benachrichtigung an die ursprüngliche statt an die aktuelle Ärztin ging und dass das Audit-Protokoll nur eine der beiden Änderungen erfasste. Nichts davon war von irgendeinem Test abgedeckt, weil das Szenario der gleichzeitigen Bearbeitung nie jemandem eingefallen war, der Tests schrieb. Alle drei wurden nach der Behebung zu automatisierten Tests. Über zwei Quartale produzierten die Sitzungen 61 Befunde, von denen 34 als Defekte akzeptiert und 9 als bedeutend eingestuft wurden.

Die Klassifizierungsmehrdeutigkeit erwies sich als aufschlussreich in ihrem eigenen Recht. Vierzehn Befunde waren Verhaltensweisen, die das Team nicht sicher als Defekte bezeichnen konnte — ein Termintyp, der überlappende Buchungen erlaubte, ein Stornierungsfenster, das an Monatsgrenzen anders funktionierte, ein Statusübergang, den das Zustandsdiagramm verbot. Diese dem klinischen Betriebsteam vorzulegen enthüllte, dass elf beabsichtigte, undokumentierte Anpassungen an echten klinischen Arbeitsabläufen waren, Jahre zuvor etabliert und nur langjährigem Personal bekannt. Diese elf wurden dokumentiert und in Characterization Tests umgewandelt, die sie davor schützten, von einem künftigen Entwickler, der das Zustandsdiagramm liest, „repariert" zu werden. Die verbleibenden drei waren echte Defekte, die seit Jahren still Terminierungsprobleme verursacht hatten.
